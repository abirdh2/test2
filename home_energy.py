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
import hmac
from io import BytesIO
import zipfile
from datetime import date
import numpy as np

st.set_page_config(
    initial_sidebar_state="expanded",
)

st.title("⚡ Home Energy Consumption and Cost Analyzer")
st.subheader("Ha'sharon 24 St. Apt 30")

try:
    stored_secret = st.secrets["pass"]["app_password"]
except KeyError:
    st.error("🚨 Configuration Error: The secret 'pass.app_password' was not found in secrets.toml.")
    st.stop()
    
entered_password = st.text_input("🔑 Enter password", type="password")

if entered_password: 
    if not hmac.compare_digest(entered_password, stored_secret):
        st.error("😕 Password incorrect. Access denied.")
        st.stop()
else:
    st.stop()

CHECK_MISSINIG_HOURS = True

jewish_holidays = [
    '2025-10-03', '2025-10-12', '2025-10-17', '2025-12-30',
    '2026-03-03', '2026-04-02', '2026-04-03', '2026-04-08', '2026-04-09', '2026-05-22', '2026-05-23', '2026-07-23', '2026-09-12', '2026-09-13', '2026-09-14', '2026-09-21', '2026-09-26', '2026-09-27', '2026-10-03', '2026-10-04',
    '2027-01-07', '2027-03-23', '2027-04-22', '2027-04-23', '2027-04-28', '2027-04-29', '2027-06-11', '2027-06-12', '2027-08-12', '2027-10-02', '2027-10-03', '2027-10-04', '2027-10-11', '2027-10-16', '2027-10-17', '2027-10-23', '2027-10-24'
]

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(
    st.secrets["gcp"],
    scopes=scope
)

gc = gspread.authorize(creds)

spreadsheet = gc.open('home_energy')
worksheet = spreadsheet.get_worksheet_by_id(0)

data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
csv_output = df.to_csv(index=False)

st.download_button(
    label="⬇️ Download raw data as CSV",
    data=csv_output,
    file_name='car_energy_raw_data_pandas.csv',
    mime='text/csv',
)

df["received_at_utc"] = pd.to_datetime(df["received_at_utc"])
df["received_at_local"] = df["received_at_utc"].dt.tz_convert("Asia/Jerusalem")
df["received_at_local_naive"] = df["received_at_local"].dt.tz_localize(None)

conn = sqlite3.connect("energy_data.db")
df.to_sql("raw_data", conn, if_exists="replace", index=False)

conn.execute("DROP VIEW IF EXISTS hourly_deltas")
conn.execute("""
CREATE VIEW hourly_deltas AS
WITH hourly_snapshots AS (
    SELECT 
        device_id,
        received_at_local_naive,
        received_at_local,
        received_at_utc,
        total_wh,
        ROW_NUMBER() OVER (
            PARTITION BY device_id, DATE(received_at_local_naive), STRFTIME('%H', received_at_local_naive) 
            ORDER BY received_at_local_naive DESC
        ) as rank
    FROM raw_data
)
SELECT
    device_id,
    CASE
        WHEN CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) = 0
        THEN DATE(received_at_local_naive, '-1 day')
        ELSE DATE(received_at_local_naive)
    END AS date_local,
    (CAST(STRFTIME('%H', received_at_local_naive) AS INTEGER) - 1 + 24) % 24 AS hour_local,
    received_at_local,
    total_wh,
    LAG(total_wh) OVER (PARTITION BY device_id ORDER BY received_at_local) AS prev_wh,
    (total_wh - LAG(total_wh) OVER (PARTITION BY device_id ORDER BY received_at_local)) / 1000.0 AS delta_kwh
FROM hourly_snapshots
WHERE rank = 1
""")

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

def check_missing_hours(conn, view_name="hourly_deltas", check_type="Consumption Interval Completeness"):
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
    if not missing_hours_df.empty:
        for _, row in missing_hours_df.iterrows():
            present_hours = set(map(int, row['hours_list'].split(',')))
            all_hours = set(range(24))
            missing = sorted(list(all_hours - present_hours))
        return False
    return True

if CHECK_MISSINIG_HOURS:
  check_missing_hours(conn, view_name="raw_hourly_presence", check_type="Raw Data Point Presence")

holidays_df = pd.DataFrame(jewish_holidays, columns=['holiday_date'])
holidays_df['is_holiday'] = 1

with sqlite3.connect("energy_data.db") as conn_holidays:
    holidays_df.to_sql("holidays", conn_holidays, if_exists="replace", index=False)

st.sidebar.markdown("### 📅 Date Range")

START_DATE = st.sidebar.date_input("Start Date", date.today().replace(day=1), key="start_date_input")
END_DATE = st.sidebar.date_input("End Date", date.today(), key="end_date_input")

start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

if start_dt.year != end_dt.year or start_dt.month != end_dt.month:
    st.error("🚨 **Selection Error:** The selected date range must be within the same calendar month.")
    st.stop()
else:
    st.sidebar.success("✅ Dates valid.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Billing Configuration")

billing_plan = st.sidebar.selectbox(
    "Select Electricity Plan",
    (
        "Option 1: Fixed Price", 
        "Option 2: Day Saver (07:00-17:00)", 
        "Option 3: Night Saver (23:00-07:00)"
    )
)

# Regular price:
full_price = 0.6432
fixed_discount_price = full_price * 0.935
day_discount_price = full_price * 0.84
night_discount_price = full_price * 0.79

# --- ADDED: Editable Tariffs ---
with st.sidebar.expander("⚙️ Adjust Tariffs (₪/kWh)"):
    if "Option 1" in billing_plan:
        rate_fixed = st.number_input("Fixed Rate", value=fixed_discount_price, step=0.0001, format="%.4f")
    elif "Option 2" in billing_plan:
        rate_day_low = st.number_input("Low Rate (Daytime)", value=day_discount_price, step=0.0001, format="%.4f")
        rate_day_high = st.number_input("High Rate (Other times)", value=full_price, step=0.0001, format="%.4f")
    elif "Option 3" in billing_plan:
        rate_night_low = st.number_input("Low Rate (Nighttime)", value=night_discount_price, step=0.01, format="%.4f")
        rate_night_high = st.number_input("High Rate (Other times)", value=full_price, step=0.0001, format="%.4f")

with sqlite3.connect("energy_data.db") as conn_hourly:
    df_calc = pd.read_sql_query("SELECT * FROM hourly_deltas", conn_hourly)

df_calc['date_local'] = pd.to_datetime(df_calc['date_local'])
df_calc['hour_local'] = pd.to_numeric(df_calc['hour_local'])

df_calc = df_calc[
    (df_calc['date_local'] >= pd.to_datetime(START_DATE)) & 
    (df_calc['date_local'] <= pd.to_datetime(END_DATE))
].copy()

df_calc['weekday'] = df_calc['date_local'].dt.weekday
is_sun_to_thu_mask = df_calc['weekday'].isin([6, 0, 1, 2, 3])

df_calc['rate_applied'] = 0.0
df_calc['is_cheap_rate'] = False 

if "Option 1" in billing_plan:
    df_calc['rate_applied'] = rate_fixed
    df_calc['is_cheap_rate'] = True 
elif "Option 2" in billing_plan:
    condition = (is_sun_to_thu_mask) & (df_calc['hour_local'] >= 7) & (df_calc['hour_local'] < 17)
    df_calc['rate_applied'] = np.where(condition, rate_day_low, rate_day_high)
    df_calc['is_cheap_rate'] = condition
elif "Option 3" in billing_plan:
    condition = (is_sun_to_thu_mask) & ((df_calc['hour_local'] >= 23) | (df_calc['hour_local'] < 7))
    df_calc['rate_applied'] = np.where(condition, rate_night_low, rate_night_high)
    df_calc['is_cheap_rate'] = condition

df_calc['cost'] = df_calc['delta_kwh'] * df_calc['rate_applied']

daily_summary_filtered = df_calc.groupby('date_local').agg(
    total_kwh=('delta_kwh', 'sum'),
    total_cost=('cost', 'sum'),
    kwh_cheap=('delta_kwh', lambda x: x[df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    kwh_expensive=('delta_kwh', lambda x: x[~df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    cost_cheap=('cost', lambda x: x[df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    cost_expensive=('cost', lambda x: x[~df_calc.loc[x.index, 'is_cheap_rate']].sum())
).reset_index()

total_kwh_overall = daily_summary_filtered['total_kwh'].sum()
total_peak_kwh = daily_summary_filtered['kwh_expensive'].sum()
total_off_peak_kwh = daily_summary_filtered['kwh_cheap'].sum()
total_cost = daily_summary_filtered['total_cost'].sum()
cost_total_peak_kwh = daily_summary_filtered['cost_expensive'].sum()
cost_total_off_peak_kwh = daily_summary_filtered['cost_cheap'].sum()

daily_summary_df = daily_summary_filtered.copy()
daily_summary_df['plotted_kwh_peak'] = daily_summary_df['kwh_expensive'] 
daily_summary_df['plotted_kwh_off_peak'] = daily_summary_df['kwh_cheap']

peak_label = 'Standard/High Rate kWh'
off_peak_label = 'Discounted/Low Rate kWh'
    
summary_data = {
    'Metric': [
        'Total Overall kWh',
        'Total High/Standard Rate kWh',
        'Total Discounted Rate kWh',
        'Cost for High Rate',
        'Cost for Discounted Rate',
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

st.subheader("Consumption and Cost Summary")
st.dataframe(summary_df)

if 'date_local' in daily_summary_df.columns:
    daily_summary_df['date_local'] = pd.to_datetime(daily_summary_df['date_local'], errors='coerce')
    daily_summary_df = daily_summary_df.reset_index(drop=True)

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
ax.set_title('Daily Energy Consumption: Rate Breakdown', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(daily_summary_df['date_local'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax.legend()
max_total_kwh = daily_summary_df['total_kwh'].max() * 1.15 if not daily_summary_df.empty else 10
ax.set_ylim(0, max_total_kwh)

for i, row in daily_summary_df.iterrows():
    if row['plotted_kwh_off_peak'] > 0:
        ax.text(x_pos[i], row['plotted_kwh_off_peak']/2, f"{row['plotted_kwh_off_peak']:.2f}", ha='center', va='center', color='darkgreen', fontsize=8)
    if row['plotted_kwh_peak'] > 0:
        ax.text(x_pos[i], row['plotted_kwh_off_peak'] + row['plotted_kwh_peak']/2, f"{row['plotted_kwh_peak']:.2f}", ha='center', va='center', color='darkred', fontsize=8)
    if row['total_kwh'] > 0:
        ax.text(x_pos[i], row['total_kwh']+0.5, f"{row['total_kwh']:.2f}", ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
data_kwh = [total_peak_kwh, total_off_peak_kwh]
combined_total_kwh = sum(data_kwh)
ax.bar(['High Rate kWh', 'Low Rate kWh'], data_kwh, color=['lightcoral', 'lightgreen'])
ax.set_ylabel('Total Energy (kWh)')
ax.set_title('Energy Consumption by Rate Type')
for i, v in enumerate(data_kwh):
    ax.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
y_max_kwh = combined_total_kwh * 1.15 if combined_total_kwh > 0 else 5
ax.set_ylim(0, y_max_kwh)
ax.text(0.5, combined_total_kwh + (y_max_kwh * 0.05), f"Total: {combined_total_kwh:.2f} kWh", ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

if st.checkbox('Show daily Graphs'):
    hourly_df_plot = df_calc.copy()
    unique_dates_to_plot = hourly_df_plot['date_local'].dt.normalize().unique()
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for date_val in sorted(unique_dates_to_plot):
            date_normalized = pd.to_datetime(date_val).normalize()
            daily_hourly_data = hourly_df_plot[hourly_df_plot['date_local'].dt.normalize() == date_normalized].copy()
            daily_hourly_data = daily_hourly_data.sort_values(by='hour_local')
            daily_hourly_data['kwh_plot_cheap'] = np.where(daily_hourly_data['is_cheap_rate'], daily_hourly_data['delta_kwh'], 0)
            daily_hourly_data['kwh_plot_expensive'] = np.where(~daily_hourly_data['is_cheap_rate'], daily_hourly_data['delta_kwh'], 0)
            daily_hourly_data.fillna(0, inplace=True)

            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos_h = daily_hourly_data['hour_local']
            ax.bar(x_pos_h, daily_hourly_data['kwh_plot_cheap'], color='lightgreen', width=0.8, edgecolor='white', label='Discount Rate')
            ax.bar(x_pos_h, daily_hourly_data['kwh_plot_expensive'], bottom=daily_hourly_data['kwh_plot_cheap'], color='lightcoral', width=0.8, edgecolor='white', label='High Rate')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('kWh', fontsize=12)
            ax.set_title(f'Hourly: {date_normalized.strftime("%Y-%m-%d")}', fontsize=14)
            ax.set_xticks(range(24))
            ax.legend()
            max_kwh_day = daily_hourly_data['delta_kwh'].max()
            ax.set_ylim(0, max_kwh_day * 1.15 if max_kwh_day > 0 else 1)
            for i, (hour, total_kwh) in enumerate(daily_hourly_data[['hour_local', 'delta_kwh']].values):
                if total_kwh > 0:
                    ax.text(hour, total_kwh + (max_kwh_day * 0.02), f"{total_kwh:.2f}", ha='center', va='bottom', color='black', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plot_buffer = BytesIO()
            fig.savefig(plot_buffer, format="png")
            plot_buffer.seek(0)
            zip_file.writestr(f'hourly_{date_normalized.strftime("%Y-%m-%d")}.png', plot_buffer.read())
            plt.close(fig)

    st.markdown("---")
    st.download_button(
        label="⬇️ Download All Plots as ZIP",
        data=zip_buffer.getvalue(),
        file_name=f'plots_{START_DATE}_{END_DATE}.zip',
        mime="application/zip"
    )

if 'conn' in locals() and conn:
    conn.close()
