import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import sqlite3
from zoneinfo import ZoneInfo
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs

# Login
password = st.text_input("Enter password", type="password")
if password != "1112":
    st.stop()

# --- Define Date Window Parameters, Price Rates, and Holidays ---
START_DATE = '2025-11-01'
END_DATE = '2025-11-30'

WEEKDAY_PEAK_RATE_DEFAULT = 0.4977
WEEKDAY_OFFPEAK_RATE_DEFAULT = 0.446
WEEKEND_PEAK_RATE_DEFAULT = 0.4977
WEEKEND_OFFPEAK_RATE_DEFAULT = 0.446
WEEKEND_HAS_PEAK_RATE = False # Explicitly set as False
CHECK_MISSINIG_HOURS = True

jewish_holidays = [
    '2025-10-03', '2025-10-12', '2025-10-17', '2025-12-30',
    '2026-03-03', '2026-04-02', '2026-04-03', '2026-04-08', '2026-04-09', '2026-05-22', '2026-05-23', '2026-07-23', '2026-09-12', '2026-09-13', '2026-09-14', '2026-09-21', '2026-09-26', '2026-09-27', '2026-10-03', '2026-10-04',
    '2027-01-07', '2027-03-23', '2027-04-22', '2027-04-23', '2027-04-28', '2027-04-29', '2027-06-11', '2027-06-12', '2027-08-12', '2027-10-02', '2027-10-03', '2027-10-04', '2027-10-11', '2027-10-16', '2027-10-17', '2027-10-23', '2027-10-24'
]


# --- 0. Google Sheets Authentication ---
# Use the correct scopes
scope = [
    "https://www.googleapis.com/auth/spreadsheets",      # Full access
    "https://www.googleapis.com/auth/drive.file",        # Access files created by the service account
    "https://www.googleapis.com/auth/drive"              # Optional: access all Drive files
]

creds = Credentials.from_service_account_info(
    st.secrets["gcp"],
    scopes=scope
)

gc = gspread.authorize(creds)

# --- 1. Load data from Google Sheet and Initial Processing ---
# Replace 'Your Spreadsheet Title' with the exact name of your Google Sheet
spreadsheet = gc.open('car-energy')
worksheet = spreadsheet.get_worksheet_by_id(17794472) # Access the first sheet/tab

# Get all data as a list of lists
data = worksheet.get_all_values()

df = pd.DataFrame(data[1:], columns=data[0])
# Ensure proper data types
df["received_at_utc"] = pd.to_datetime(df["received_at_utc"])

# Convert to local Jerusalem time
# print(f"Sample UTC timestamps before conversion:\n{df['received_at_utc'].head()}") # Commented out logging print
df["received_at_local"] = df["received_at_utc"].dt.tz_convert("Asia/Jerusalem")

# Remove tzinfo for SQLite (naive datetime)
df["received_at_local_naive"] = df["received_at_local"].dt.tz_localize(None)
# print(f"Sample local timestamps after conversion:\n{df['received_at_local'].head()}") # Commented out logging print

# --- 2. Create SQLite database and raw_data table ---
conn = sqlite3.connect("energy_data.db")
df.to_sql("raw_data", conn, if_exists="replace", index=False)


# --- 3. Compute hourly deltas view ---
conn.execute("DROP VIEW IF EXISTS hourly_deltas")
conn.execute("""
CREATE VIEW hourly_deltas AS
SELECT
    device_id,
    CASE
        WHEN CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) = 0
        THEN DATE(received_at_local_naive, '-1 day')
        ELSE DATE(received_at_local_naive)
    END AS date_local,
    (CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) - 1 + 24) % 24 AS hour_local,
    received_at_local,
    received_at_utc,
    STRFTIME('%H', received_at_utc) AS hour_utc,
    total_wh,
    LAG(total_wh) OVER (PARTITION BY device_id ORDER BY received_at_local) AS prev_wh,
    (total_wh - LAG(total_wh) OVER (PARTITION BY device_id ORDER BY received_at_local)) / 1000.0 AS delta_kwh,
    (CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) - 1 + 24) % 24 AS prev_hour_local_original
FROM raw_data
""")

# --- NEW: View for Raw Hourly Presence Check ---
# This view identifies if a raw measurement exists for each hour, regardless of delta calculation
conn.execute("DROP VIEW IF EXISTS raw_hourly_presence")
conn.execute("""
CREATE VIEW raw_hourly_presence AS
SELECT
    device_id,
    DATE(received_at_local_naive) AS date_local,
    CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) AS hour_local
FROM raw_data
GROUP BY device_id, DATE(received_at_local_naive), CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER)
""")

# --- 3.5. Test for missing hours ---
def check_missing_hours(conn, view_name="hourly_deltas", check_type="Consumption Interval Completeness"):
    # print(f"\n=== Checking for Missing Hours ({check_type}) ===")
    hourly_check_query = f"""
    SELECT
        device_id,
        date_local,
        COUNT(DISTINCT hour_local) as hours_present,
        GROUP_CONCAT(DISTINCT hour_local) as hours_list
    FROM {view_name}
    GROUP BY device_id, date_local
    HAVING hours_present < 24
    ORDER BY device_id, date_local
    """
    missing_hours_df = pd.read_sql_query(hourly_check_query, conn)
    if missing_hours_df.empty:
        print(f"\u2705 No missing hours detected in {check_type}. All days have complete 24-hour data.")
        return True
    else:
        print(f"\u26a0\ufe0f  Found {len(missing_hours_df)} day(s) with missing hours in {check_type}:")
        # print(missing_hours_df)
        for _, row in missing_hours_df.iterrows():
            present_hours = set(map(int, row['hours_list'].split(',')))
            all_hours = set(range(24))
            missing = sorted(list(all_hours - present_hours))
            print(f"\n  Date: {row['date_local']}")
            print(f"  Missing hours: {', '.join([f'{h:02d}' for h in missing])}")
        return False

if CHECK_MISSINIG_HOURS:
  # check_missing_hours(conn, view_name="hourly_deltas", check_type="Consumption Interval Completeness")
  check_missing_hours(conn, view_name="raw_hourly_presence", check_type="Raw Data Point Presence")

# --- 4. Daily summary view ---
conn.execute("DROP VIEW IF EXISTS daily_summary")
conn.execute("""
CREATE VIEW daily_summary AS
SELECT
    device_id,
    date_local,
    ROUND(SUM(CASE WHEN delta_kwh > 0 THEN delta_kwh ELSE 0 END), 6) AS total_kwh,
    ROUND(SUM(CASE
        WHEN hour_local BETWEEN 17 AND 21 THEN delta_kwh ELSE 0 END), 6) AS kwh_17_22,
    ROUND(SUM(CASE
        WHEN hour_local NOT BETWEEN 17 AND 21 THEN delta_kwh ELSE 0 END), 6) AS kwh_22_17
FROM hourly_deltas
GROUP BY device_id, date_local
""")

holidays_df = pd.DataFrame(jewish_holidays, columns=['holiday_date'])
holidays_df['is_holiday'] = 1

with sqlite3.connect("energy_data.db") as conn_holidays:
    holidays_df.to_sql("holidays", conn_holidays, if_exists="replace", index=False)

holidays_series = pd.to_datetime(pd.Series(jewish_holidays)) # Defined once globally

# Fetch daily_summary data for plotting and cost calculations
with sqlite3.connect("energy_data.db") as conn_summary:
    report_daily_summary = pd.read_sql_query("SELECT * FROM daily_summary", conn_summary)


# --- 6. Calculate Aggregated Consumption and Costs ---
base_query_parts = []
cost_columns_to_sum = []

base_query_parts.append(f"""
    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) NOT IN (5, 6) AND
             T1.date_local NOT IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22
        ELSE 0
    END) AS weekday_kwh_17_22,

    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) NOT IN (5, 6) AND
             T1.date_local NOT IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22
        ELSE 0
    END) * {WEEKDAY_PEAK_RATE_DEFAULT} AS cost_weekday_17_22,


    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) NOT IN (5, 6) AND
             T1.date_local NOT IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_22_17
        ELSE 0
    END) AS weekday_kwh_22_17,

    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) NOT IN (5, 6) AND
             T1.date_local NOT IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_22_17
        ELSE 0
    END) * {WEEKDAY_OFFPEAK_RATE_DEFAULT} AS cost_weekday_22_17
""")
cost_columns_to_sum.extend(['cost_weekday_17_22', 'cost_weekday_22_17'])


if WEEKEND_HAS_PEAK_RATE:
    base_query_parts.append(f"""
    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22
        ELSE 0
    END) AS weekend_holiday_kwh_17_22,

    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22
        ELSE 0
    END) * {WEEKEND_PEAK_RATE_DEFAULT} AS cost_weekend_holiday_17_22,


    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_22_17
        ELSE 0
    END) AS weekend_holiday_kwh_22_17,

    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_22_17
        ELSE 0
    END) * {WEEKEND_OFFPEAK_RATE_DEFAULT} AS cost_weekend_holiday_22_17
""")
    cost_columns_to_sum.extend(['cost_weekend_holiday_17_22', 'cost_weekend_holiday_22_17'])
else:
    base_query_parts.append(f"""
    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22 + T1.kwh_22_17
        ELSE 0
    END) AS weekend_holiday_kwh_off_peak,

    SUM(CASE
        WHEN CAST(STRFTIME('%w', T1.date_local) AS INTEGER) IN (5, 6) OR
             T1.date_local IN (SELECT holiday_date FROM holidays)
        THEN T1.kwh_17_22 + T1.kwh_22_17
        ELSE 0
    END) * {WEEKEND_OFFPEAK_RATE_DEFAULT} AS cost_weekend_holiday_off_peak
""")
    cost_columns_to_sum.append('cost_weekend_holiday_off_peak')

query = f"""
SELECT
    {','.join(base_query_parts)}
FROM daily_summary T1
WHERE T1.date_local BETWEEN '{START_DATE}' AND '{END_DATE}'
"""

with sqlite3.connect("energy_data.db") as conn_report:
    report = pd.read_sql_query(query, conn_report)
# print("--- Consumption and Total Cost Summary ---") # Commented out logging print
# print(report) # Commented out logging print

report['Total_Cost'] = sum(report[col] for col in cost_columns_to_sum)

# print("\n--- Grand Total Cost (Aggregated) ---") # Commented out logging print
# print(f"Total Consumption Cost: {report['Total_Cost'].sum():.2f}") # Commented out logging print

# --- 7. Generate Daily Summary Plot ---
daily_summary_df = report_daily_summary.copy()
daily_summary_df['date_local'] = pd.to_datetime(daily_summary_df['date_local'])

daily_summary_df['plotted_kwh_peak'] = daily_summary_df['kwh_17_22']
daily_summary_df['plotted_kwh_off_peak'] = daily_summary_df['kwh_22_17']

is_weekend = (daily_summary_df['date_local'].dt.weekday == 4) | (daily_summary_df['date_local'].dt.weekday == 5)
is_holiday = daily_summary_df['date_local'].dt.date.isin(holidays_series.dt.date)
is_weekend_or_holiday = is_weekend | is_holiday


# --- Sidebar filters ---
st.sidebar.markdown("### 📅 Date")

START_DATE = st.sidebar.date_input(
    "Start Date", 
    daily_summary_df['date_local'].min(),
    key="start_date_input"
)
END_DATE = st.sidebar.date_input(
    "End Date", 
    daily_summary_df['date_local'].max(),
    key="end_date_input"
)

st.sidebar.markdown("### ⚡ Electricity Rates")

WEEKDAY_PEAK_RATE = st.sidebar.number_input(
    "Weekday Peak Rate (₪/kWh)",
    min_value=0.0,
    max_value=5.0,
    value=WEEKDAY_PEAK_RATE_DEFAULT,
    step=0.0001,
    format="%.4f",   # 👈 prevents rounding
    key="weekday_peak_rate"
)

WEEKDAY_OFFPEAK_RATE = st.sidebar.number_input(
    "Weekday Off-Peak Rate (₪/kWh)",
    min_value=0.0,
    max_value=5.0,
    value=WEEKDAY_OFFPEAK_RATE_DEFAULT,
    step=0.0001,
    format="%.4f",
    key="weekday_offpeak_rate"
)

WEEKEND_PEAK_RATE = st.sidebar.number_input(
    "Weekend Peak Rate (₪/kWh)",
    min_value=0.0,
    max_value=5.0,
    value=WEEKEND_PEAK_RATE_DEFAULT,
    step=0.0001,
    format="%.4f",
    key="weekend_peak_rate"
)

WEEKEND_OFFPEAK_RATE = st.sidebar.number_input(
    "Weekend Off-Peak Rate (₪/kWh)",
    min_value=0.0,
    max_value=5.0,
    value=WEEKEND_OFFPEAK_RATE_DEFAULT,
    step=0.0001,
    format="%.4f",
    key="weekend_offpeak_rate"
)

WEEKEND_HAS_PEAK_RATE = st.sidebar.checkbox(
    "Weekend has peak rate", 
    value=False,
    key="weekend_peak_checkbox"
)


# --- AFTER all st.sidebar.number_input widgets are defined ---

# 1. Re-fetch and filter the daily summary using the current sidebar dates
with sqlite3.connect("energy_data.db") as conn_summary:
    report_daily_summary = pd.read_sql_query("SELECT * FROM daily_summary", conn_summary)
    
report_daily_summary['date_local'] = pd.to_datetime(report_daily_summary['date_local'])

# Filter by the Streamlit sidebar dates
daily_summary_filtered = report_daily_summary[
    (report_daily_summary['date_local'] >= pd.to_datetime(START_DATE)) &
    (report_daily_summary['date_local'] <= pd.to_datetime(END_DATE))
].copy()

# 2. Determine day types and consumption allocations based on sidebar checkbox
holidays_series = pd.to_datetime(pd.Series(jewish_holidays)).dt.normalize()
daily_summary_filtered['is_weekend'] = (daily_summary_filtered['date_local'].dt.weekday == 4) | (daily_summary_filtered['date_local'].dt.weekday == 5)
daily_summary_filtered['is_holiday'] = daily_summary_filtered['date_local'].dt.normalize().isin(holidays_series)
daily_summary_filtered['is_weekend_or_holiday'] = daily_summary_filtered['is_weekend'] | daily_summary_filtered['is_holiday']

# Initialize columns for cost calculation
daily_summary_filtered['kwh_weekday_peak'] = 0.0
daily_summary_filtered['kwh_weekday_offpeak'] = 0.0
daily_summary_filtered['kwh_weekend_peak'] = 0.0
daily_summary_filtered['kwh_weekend_offpeak'] = 0.0

# Logic to distribute kWh based on day type and weekend peak checkbox
is_weekday = ~daily_summary_filtered['is_weekend_or_holiday']
is_special_day = daily_summary_filtered['is_weekend_or_holiday']

# Weekday allocation
daily_summary_filtered.loc[is_weekday, 'kwh_weekday_peak'] = daily_summary_filtered.loc[is_weekday, 'kwh_17_22']
daily_summary_filtered.loc[is_weekday, 'kwh_weekday_offpeak'] = daily_summary_filtered.loc[is_weekday, 'kwh_22_17']

# Weekend/Holiday allocation
if WEEKEND_HAS_PEAK_RATE:
    daily_summary_filtered.loc[is_special_day, 'kwh_weekend_peak'] = daily_summary_filtered.loc[is_special_day, 'kwh_17_22']
    daily_summary_filtered.loc[is_special_day, 'kwh_weekend_offpeak'] = daily_summary_filtered.loc[is_special_day, 'kwh_22_17']
else:
    # Everything on weekend/holiday is off-peak
    daily_summary_filtered.loc[is_special_day, 'kwh_weekend_offpeak'] = daily_summary_filtered.loc[is_special_day, 'total_kwh']

# 3. Aggregate Consumption
total_kwh_overall = daily_summary_filtered['total_kwh'].sum()

total_kwh_weekday_peak = daily_summary_filtered['kwh_weekday_peak'].sum()
total_kwh_weekday_offpeak = daily_summary_filtered['kwh_weekday_offpeak'].sum()
total_kwh_weekend_peak = daily_summary_filtered['kwh_weekend_peak'].sum()
total_kwh_weekend_offpeak = daily_summary_filtered['kwh_weekend_offpeak'].sum()

# Aggregate Peak and Off-Peak for the Summary Table
total_peak_kwh = total_kwh_weekday_peak + total_kwh_weekend_peak
total_off_peak_kwh = total_kwh_weekday_offpeak + total_kwh_weekend_offpeak

# 4. Calculate Costs using sidebar rates
cost_weekday_peak = total_kwh_weekday_peak * WEEKDAY_PEAK_RATE
cost_weekday_offpeak = total_kwh_weekday_offpeak * WEEKDAY_OFFPEAK_RATE
cost_weekend_peak = total_kwh_weekend_peak * WEEKEND_PEAK_RATE
cost_weekend_offpeak = total_kwh_weekend_offpeak * WEEKEND_OFFPEAK_RATE

# Aggregate Costs
cost_total_peak_kwh = cost_weekday_peak + cost_weekend_peak
cost_total_off_peak_kwh = cost_weekday_offpeak + cost_weekend_offpeak
total_cost = cost_total_peak_kwh + cost_total_off_peak_kwh

# 5. Display Summary Table
# ... your summary_df creation and st.dataframe(summary_df) code here ...


if not WEEKEND_HAS_PEAK_RATE:
    daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_off_peak'] += daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_peak']
    daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_peak'] = 0

peak_label = 'Peak (17-22) kWh'
off_peak_label = 'Off-Peak (22-17) kWh'
if not WEEKEND_HAS_PEAK_RATE:
    peak_label = 'Peak (Weekday Only) kWh'
    off_peak_label = 'Off-Peak (Includes Weekend/Holiday) kWh'
    

# --- Aggregated Consumption and Cost Summary (Visualized) ---
summary_data = {
    'Metric': [
        'Total Overall kWh',
        'Total Peak kWh',
        'Total Off-Peak kWh',
        'Cost for Total Peak kWh',
        'Cost for Total Off-Peak kWh',
        'Grand Total Cost'
    ],
    'Value': [
        f"{total_kwh_overall:.2f} kWh",
        f"{total_peak_kwh:.2f} kWh",
        f"{total_off_peak_kwh:.2f} kWh",
        f"₪ {cost_total_peak_kwh:.2f}",
        f"₪ {cost_total_off_peak_kwh:.2f}",
        f"₪ {total_cost:.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)

print("\n--- Aggregated Consumption and Cost Summary ---")
print(summary_df)


# --- Ensure 'date_local' exists and is datetime ---
if 'date_local' not in daily_summary_df.columns:
    st.error("Error: 'date_local' column not found in daily_summary_df!")
else:
    daily_summary_df['date_local'] = pd.to_datetime(daily_summary_df['date_local'], errors='coerce')
    daily_summary_df = daily_summary_df[
        (daily_summary_df['date_local'] >= pd.to_datetime(START_DATE)) &
        (daily_summary_df['date_local'] <= pd.to_datetime(END_DATE))
    ].copy()

    # Reset index so x_pos matches iterrows
    daily_summary_df = daily_summary_df.reset_index(drop=True)

# --- Display Summary Table ---
summary_data = {
    'Metric': [
        'Total Overall kWh',
        'Total Peak kWh',
        'Total Off-Peak kWh',
        'Cost for Total Peak kWh',
        'Cost for Total Off-Peak kWh',
        'Grand Total Cost'
    ],
    'Value': [
        f"{total_kwh_overall:.2f} kWh",
        f"{total_peak_kwh:.2f} kWh",
        f"{total_off_peak_kwh:.2f} kWh",
        f"₪ {cost_total_peak_kwh:.2f}",
        f"₪ {cost_total_off_peak_kwh:.2f}",
        f"₪ {total_cost:.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
st.subheader("Aggregated Consumption and Cost Summary")
st.dataframe(summary_df)

# --- Daily Summary Plot ---
off_peak_label = 'Off-Peak kWh'
peak_label = 'Peak kWh'

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = range(len(daily_summary_df))
width = 0.6

ax.bar(x_pos, daily_summary_df['plotted_kwh_off_peak'], color='lightgreen', width=width, edgecolor='white', label=off_peak_label)
peak_mask = daily_summary_df['plotted_kwh_peak'] > 0
if peak_mask.any():
    ax.bar(daily_summary_df.index[peak_mask],
           daily_summary_df.loc[peak_mask, 'plotted_kwh_peak'],
           bottom=daily_summary_df.loc[peak_mask, 'plotted_kwh_off_peak'],
           color='lightcoral', width=width, edgecolor='white', label=peak_label)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax.set_title('Daily Energy Consumption: Peak and Off-Peak', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(daily_summary_df['date_local'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i, row in daily_summary_df.iterrows():
    if row['plotted_kwh_off_peak'] > 0:
        ax.text(x_pos[i], row['plotted_kwh_off_peak']/2, f"{row['plotted_kwh_off_peak']:.2f}", ha='center', va='center', color='darkgreen', fontsize=8)
    if row['plotted_kwh_peak'] > 0:
        ax.text(x_pos[i], row['plotted_kwh_off_peak'] + row['plotted_kwh_peak']/2, f"{row['plotted_kwh_peak']:.2f}", ha='center', va='center', color='darkred', fontsize=8)
    if row['total_kwh'] > 0:
        ax.text(x_pos[i], row['total_kwh']+0.5, f"{row['total_kwh']:.2f}", ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')

plt.tight_layout()
folder_daily = f"from_{START_DATE}_to_{END_DATE}"
if not path.exists(folder_daily):
    makedirs(folder_daily)
plt.savefig(f'{folder_daily}/daily_energy_consumption.png')
st.pyplot(fig)
plt.close(fig)

############

# --- Load hourly data from SQLite ---
with sqlite3.connect("energy_data.db") as conn_hourly:
    hourly_df = pd.read_sql_query("SELECT * FROM hourly_deltas", conn_hourly)

# Ensure date_local is datetime and hour_local numeric
hourly_df['date_local'] = pd.to_datetime(hourly_df['date_local'])
hourly_df['hour_local'] = pd.to_numeric(hourly_df['hour_local'], errors='coerce')

# Filter by selected date range
hourly_df_filtered = hourly_df[
    (hourly_df['date_local'] >= pd.to_datetime(START_DATE)) &
    (hourly_df['date_local'] <= pd.to_datetime(END_DATE))
].copy()

# Limit to first 31 unique days if needed
unique_days = hourly_df_filtered['date_local'].dt.normalize().unique()
if len(unique_days) > 31:
    days_to_keep = pd.Series(unique_days).sort_values().head(31).tolist()
    hourly_df_plot = hourly_df_filtered[hourly_df_filtered['date_local'].dt.normalize().isin(days_to_keep)].copy()
else:
    hourly_df_plot = hourly_df_filtered.copy()

# --- Normalize holiday series once ---
holidays_normalized = holidays_series.dt.normalize()

# --- Generate Hourly Plots ---
unique_dates_to_plot = hourly_df_plot['date_local'].dt.normalize().unique()

for date in sorted(unique_dates_to_plot):
    date_normalized = pd.to_datetime(date).normalize()
    daily_hourly_data = hourly_df_plot[hourly_df_plot['date_local'].dt.normalize() == date_normalized].copy()
    daily_hourly_data = daily_hourly_data.sort_values(by='hour_local')

    # Determine weekend and holiday
    is_current_day_weekend = (date.weekday() >= 5)
    is_current_day_holiday = date_normalized in holidays_normalized.values
    is_current_day_weekend_or_holiday = is_current_day_weekend or is_current_day_holiday

    # Calculate peak/off-peak kWh
    if is_current_day_weekend_or_holiday and not WEEKEND_HAS_PEAK_RATE:
        daily_hourly_data['kwh_peak'] = 0
        daily_hourly_data['kwh_off_peak'] = daily_hourly_data['delta_kwh']
    else:
        daily_hourly_data['kwh_peak'] = daily_hourly_data.apply(
            lambda row: row['delta_kwh'] if 17 <= row['hour_local'] < 22 else 0, axis=1)
        daily_hourly_data['kwh_off_peak'] = daily_hourly_data['delta_kwh'] - daily_hourly_data['kwh_peak']

    daily_hourly_data.fillna(0, inplace=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    x_pos = daily_hourly_data['hour_local']
    width = 0.8

    peak_label_current_day = 'Peak (17-22) kWh'
    off_peak_label_current_day = 'Off-Peak (22-17) kWh'
    if is_current_day_weekend_or_holiday and not WEEKEND_HAS_PEAK_RATE:
        peak_label_current_day = 'Peak (Weekday Only) kWh'
        off_peak_label_current_day = 'Off-Peak (Includes Weekend/Holiday) kWh'

    ax.bar(x_pos, daily_hourly_data['kwh_off_peak'], color='lightgreen', width=width, edgecolor='white', label=off_peak_label_current_day)
    ax.bar(x_pos, daily_hourly_data['kwh_peak'], bottom=daily_hourly_data['kwh_off_peak'], color='lightcoral', width=width, edgecolor='white', label=peak_label_current_day)

    ax.set_xlabel('Hour of Day (Local Time)', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)

    title_suffix = ''
    if is_current_day_weekend_or_holiday:
        title_suffix = ' (Weekend/Holiday Treated as Off-Peak)' if not WEEKEND_HAS_PEAK_RATE else ' (Weekend/Holiday with Peak Rates)'
    ax.set_title(f'Hourly Energy Consumption for {date_normalized.strftime("%Y-%m-%d")}{title_suffix}', fontsize=14)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    max_kwh_for_day = daily_hourly_data['delta_kwh'].max()
    ax.set_ylim(0, max_kwh_for_day * 1.15 if max_kwh_for_day > 0 else 1)

    # Annotate bars
    for i, (hour, total_kwh) in enumerate(daily_hourly_data[['hour_local', 'delta_kwh']].values):
        if total_kwh > 0:
            ax.text(hour, total_kwh + (max_kwh_for_day * 0.02), f"{total_kwh:.2f}", ha='center', va='bottom', color='black', fontsize=8)

    plt.tight_layout()
    folder_hourly = f"from_{START_DATE}_to_{END_DATE}"
    if not path.exists(folder_hourly):
        makedirs(folder_hourly)
    plt.savefig(f'{folder_hourly}/hourly_consumption_{date_normalized.strftime("%Y-%m-%d")}.png')

    st.pyplot(fig)
    plt.close(fig)

# --- Sidebar: Date Range ---
with sqlite3.connect("energy_data.db") as conn:
    daily_summary_df = pd.read_sql_query("SELECT * FROM daily_summary", conn)
daily_summary_df['date_local'] = pd.to_datetime(daily_summary_df['date_local'])

START_DATE = st.sidebar.date_input("Start Date", daily_summary_df['date_local'].min())
END_DATE = st.sidebar.date_input("End Date", daily_summary_df['date_local'].max())

# Filter daily summary by date
daily_summary_df = daily_summary_df[
    (daily_summary_df['date_local'] >= pd.to_datetime(START_DATE)) &
    (daily_summary_df['date_local'] <= pd.to_datetime(END_DATE))
].copy()

# --- Compute Totals ---
daily_summary_df['plotted_kwh_peak'] = daily_summary_df['kwh_17_22']
daily_summary_df['plotted_kwh_off_peak'] = daily_summary_df['kwh_22_17']

if not WEEKEND_HAS_PEAK_RATE:
    # Treat weekends/off-holidays as off-peak
    is_weekend_or_holiday = daily_summary_df['date_local'].dt.weekday >= 5  # simple weekend flag
    daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_off_peak'] += daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_peak']
    daily_summary_df.loc[is_weekend_or_holiday, 'plotted_kwh_peak'] = 0


# --- Compute Costs based on sidebar rates ---
# cost_total_peak_kwh = total_peak_kwh * WEEKDAY_PEAK_RATE
# cost_total_off_peak_kwh = total_off_peak_kwh * WEEKDAY_OFFPEAK_RATE
# if WEEKEND_HAS_PEAK_RATE:
#     cost_total_peak_kwh += total_peak_kwh * WEEKEND_PEAK_RATE
#     cost_total_off_peak_kwh += total_off_peak_kwh * WEEKEND_OFFPEAK_RATE

# total_cost = cost_total_peak_kwh + cost_total_off_peak_kwh

# --- Visualize Totals ---
# --- kWh Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
data_kwh = [total_peak_kwh, total_off_peak_kwh]
combined_total_kwh = sum(data_kwh)

ax.bar(['Peak kWh', 'Off-Peak kWh'], data_kwh, color=['lightcoral', 'lightgreen'], label='kWh')
ax.set_ylabel('Total Energy (kWh)')
ax.set_title('Peak vs Off-Peak Energy Consumption')

# Annotate individual bars
for i, v in enumerate(data_kwh):
    ax.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

# Set Y-limit to ensure space for annotations
# Use the combined total + a buffer (e.g., 10% of the total, or a fixed amount)
y_max_kwh = combined_total_kwh * 1.15 if combined_total_kwh > 0 else 5
ax.set_ylim(0, y_max_kwh)

# Add combined total value above the bars, now with sufficient space
# The position (0.5) centers it between the two bars
ax.text(0.5, combined_total_kwh + (y_max_kwh * 0.05), # Adjusted V position
        f"Total: {combined_total_kwh:.2f} kWh", 
        ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)
plt.close(fig) # Optional: good practice to close figures

# --- Cost Plot ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
data_cost = [cost_total_peak_kwh, cost_total_off_peak_kwh]
combined_total_cost = sum(data_cost)

ax2.bar(['Peak Cost', 'Off-Peak Cost'], data_cost, color=['salmon', 'lightblue'], label='Cost')
ax2.set_ylabel('Total Cost (₪)')
ax2.set_title('Cost Breakdown: Peak vs Off-Peak')

# Annotate individual bars
for i, v in enumerate(data_cost):
    ax2.text(i, v + 0.5, f"₪ {v:.2f}", ha='center', va='bottom', fontsize=10)

# Set Y-limit to ensure space for annotations
# Use the combined total + a buffer
y_max_cost = combined_total_cost * 1.15 if combined_total_cost > 0 else 5
ax2.set_ylim(0, y_max_cost)

# Add combined total cost above the bars, now with sufficient space
ax2.text(0.5, combined_total_cost + (y_max_cost * 0.05), # Adjusted V position
         f"Total: ₪ {combined_total_cost:.2f}", 
         ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
st.pyplot(fig2)
# plt.close(fig2) # Optional: good practice to close figures

# print("Hourly plots generated for each day in the filtered range.") # Commented out logging print

# Close the initial connection
if 'conn' in locals() and conn:
    conn.close()
