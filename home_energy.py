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
import numpy as np # ### ADDED: For efficient calculation

st.set_page_config(
    # layout="wide", # Use the full width of the browser
    initial_sidebar_state="expanded",
)

st.title("⚡ Home Energy Consumption and Cost Analyzer")
st.subheader("Ha'sharon 24 St. Apt 30")

try:
    stored_secret = st.secrets["pass"]["app_password"]
except KeyError:
    # Handle the configuration error if the secret isn't found
    st.error("🚨 Configuration Error: The secret 'pass.app_password' was not found in secrets.toml.")
    st.stop()
    

# 1. Get the password the user entered
entered_password = st.text_input("🔑 Enter password", type="password")


# 2. Only proceed if the user has entered something
if entered_password: 
    
    # Securely check if the entered password DOES NOT MATCH the stored secret
    if not hmac.compare_digest(entered_password, stored_secret):
        st.error("😕 Password incorrect. Access denied.")
        st.stop() # Stop the application immediately
        
# 3. Stop the app if the input is empty (i.e., when the page first loads)
else:
    st.stop()

# ### CHANGED: Removed manual weekend config variable, logic is now inside the plan selection
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
spreadsheet = gc.open('home_energy')
worksheet = spreadsheet.get_worksheet_by_id(0) # Access the first sheet/tab

# Get all data as a list of lists
data = worksheet.get_all_values()

df = pd.DataFrame(data[1:], columns=data[0])

# Use Pandas to convert the DataFrame directly to CSV string
csv_output = df.to_csv(index=False)

# Option to download raw data:
st.download_button(
    label="⬇️ Download raw data as CSV",
    data=csv_output,
    file_name='car_energy_raw_data_pandas.csv',
    mime='text/csv',
)

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

# --- View for Raw Hourly Presence Check ---
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

# --- 4. Daily summary view (Kept for fallback, but logic moved to Python) ---
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


# --- Sidebar filters ---
st.sidebar.markdown("### 📅  Date")

START_DATE = st.sidebar.date_input(
    "Start Date", 
    date.today().replace(day=1),
    key="start_date_input"
)
END_DATE = st.sidebar.date_input(
    "End Date", 
    date.today(),
    key="end_date_input"
)
# Convert to datetime objects for easy month comparison
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)


# --- Validation Logic ---

# Check if the year or month of the two dates are different
if start_dt.year != end_dt.year or start_dt.month != end_dt.month:
    
    st.error(
        "🚨 **Selection Error:** The selected date range must be within the same calendar month. "
        "Please adjust your Start Date and End Date."
    )
    
    # Optionally, stop the rest of your app from running until the dates are corrected
    st.stop()
    
# If the validation passes, the rest of your script will execute
else:
    st.sidebar.success("✅ Dates are valid for processing.")
    # Proceed with your plotting/analysis using START_DATE and END_DATE


# -----------------------------------------------------------
# ### CHANGED: REPLACED RATE CONFIG WITH 3 PLAN OPTIONS
# -----------------------------------------------------------
st.sidebar.markdown("### ⚡ Billing Plan")

billing_plan = st.sidebar.selectbox(
    "Select Electricity Tariff",
    (
        "Option 1: Fixed Price (0.60)", 
        "Option 2: Day Saver (Sun-Thu 07:00-17:00)", 
        "Option 3: Night Saver (Sun-Thu 23:00-07:00)"
    )
)

# Display active rates info
if "Option 1" in billing_plan:
    st.sidebar.info("Rate: 0.60 ₪/kWh all day.")
elif "Option 2" in billing_plan:
    st.sidebar.info("Low Rate (0.50): Sun-Thu 07:00-17:00\n\nHigh Rate (0.70): All other times.")
elif "Option 3" in billing_plan:
    st.sidebar.info("Low Rate (0.40): Sun-Thu 23:00-07:00\n\nHigh Rate (0.70): All other times.")

# -----------------------------------------------------------


# -----------------------------------------------------------
# ### CHANGED: NEW CALCULATION LOGIC FOR 3 OPTIONS
# -----------------------------------------------------------

# 1. Load HOURLY data instead of Daily Summary
# (We need hourly data because the peak times change based on your plan)
with sqlite3.connect("energy_data.db") as conn_hourly:
    df_calc = pd.read_sql_query("SELECT * FROM hourly_deltas", conn_hourly)

# Prepare Data
df_calc['date_local'] = pd.to_datetime(df_calc['date_local'])
df_calc['hour_local'] = pd.to_numeric(df_calc['hour_local'])

# Filter by selected dates
df_calc = df_calc[
    (df_calc['date_local'] >= pd.to_datetime(START_DATE)) & 
    (df_calc['date_local'] <= pd.to_datetime(END_DATE))
].copy()

# Add Weekday info (Monday=0, Sunday=6)
df_calc['weekday'] = df_calc['date_local'].dt.weekday

# Define Israel Work Week (Sunday=6, Mon=0, Tue=1, Wed=2, Thu=3)
is_sun_to_thu_mask = df_calc['weekday'].isin([6, 0, 1, 2, 3])

# --- APPLY BILLING LOGIC PER HOUR ---

# Default values
df_calc['rate_applied'] = 0.0
df_calc['is_cheap_rate'] = False 

if "Option 1" in billing_plan:
    # Fixed Price 0.60
    df_calc['rate_applied'] = 0.60
    df_calc['is_cheap_rate'] = False # Treat all as "standard/expensive" for plotting simplicity

elif "Option 2" in billing_plan:
    # 0.50 from Sunday-Thursday between 07:00-17:00, else 0.7
    # Note: 07:00 <= hour < 17 covers 07:00 to 16:59
    condition = (is_sun_to_thu_mask) & (df_calc['hour_local'] >= 7) & (df_calc['hour_local'] < 17)
    
    df_calc['rate_applied'] = np.where(condition, 0.50, 0.70)
    df_calc['is_cheap_rate'] = condition

elif "Option 3" in billing_plan:
    # 0.40 from Sunday-Thursday between 23:00-07:00, else 0.7
    # Note: Covers 23:00-23:59 AND 00:00-06:59
    condition = (is_sun_to_thu_mask) & ((df_calc['hour_local'] >= 23) | (df_calc['hour_local'] < 7))
    
    df_calc['rate_applied'] = np.where(condition, 0.40, 0.70)
    df_calc['is_cheap_rate'] = condition

# Calculate Cost per hour
df_calc['cost'] = df_calc['delta_kwh'] * df_calc['rate_applied']

# --- AGGREGATE BACK TO DAILY SUMMARY ---
# This recreates the dataframe structure the plotting code expects
daily_summary_filtered = df_calc.groupby('date_local').agg(
    total_kwh=('delta_kwh', 'sum'),
    total_cost=('cost', 'sum'),
    # We sum up the "Cheap" (Off-peak/Discount) and "Expensive" (Standard/Peak) kWh
    kwh_cheap=('delta_kwh', lambda x: x[df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    kwh_expensive=('delta_kwh', lambda x: x[~df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    # Calculate costs for the summary table
    cost_cheap=('cost', lambda x: x[df_calc.loc[x.index, 'is_cheap_rate']].sum()),
    cost_expensive=('cost', lambda x: x[~df_calc.loc[x.index, 'is_cheap_rate']].sum())
).reset_index()

# Map columns to variables for the summary table at the bottom
total_kwh_overall = daily_summary_filtered['total_kwh'].sum()
total_peak_kwh = daily_summary_filtered['kwh_expensive'].sum() # Mapped 'Expensive' to 'Peak' for display
total_off_peak_kwh = daily_summary_filtered['kwh_cheap'].sum() # Mapped 'Cheap' to 'Off-Peak' for display
total_cost = daily_summary_filtered['total_cost'].sum()
cost_total_peak_kwh = daily_summary_filtered['cost_expensive'].sum()
cost_total_off_peak_kwh = daily_summary_filtered['cost_cheap'].sum()

# Update the DataFrame used for plotting
daily_summary_df = daily_summary_filtered.copy()
# Map to plotted columns required by subsequent code
daily_summary_df['plotted_kwh_peak'] = daily_summary_df['kwh_expensive'] 
daily_summary_df['plotted_kwh_off_peak'] = daily_summary_df['kwh_cheap']

# -----------------------------------------------------------


# ### CHANGED: Labels to reflect Generic "High Rate" vs "Low Rate"
peak_label = 'Standard/High Rate kWh'
off_peak_label = 'Discounted/Low Rate kWh'
    

# --- Aggregated Consumption and Cost Summary (Visualized) ---
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

print("\n--- Consumption and Cost Summary ---")
print(summary_df)


# --- Ensure 'date_local' exists and is datetime ---
if 'date_local' not in daily_summary_df.columns:
    st.error("Error: 'date_local' column not found in daily_summary_df!")
else:
    daily_summary_df['date_local'] = pd.to_datetime(daily_summary_df['date_local'], errors='coerce')
    
    # Reset index so x_pos matches iterrows
    daily_summary_df = daily_summary_df.reset_index(drop=True)

# --- Display Summary Table ---
st.subheader("Consumption and Cost Summary")
st.dataframe(summary_df)

# --- Daily Summary Plot ---
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
# ax.grid(axis='y', linestyle='--', alpha=0.7)

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

# --- Visualize Totals ---
# --- kWh Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
data_kwh = [total_peak_kwh, total_off_peak_kwh]
combined_total_kwh = sum(data_kwh)

ax.bar(['High Rate kWh', 'Low Rate kWh'], data_kwh, color=['lightcoral', 'lightgreen'], label='kWh')
ax.set_ylabel('Total Energy (kWh)')
ax.set_title('Energy Consumption by Rate Type')

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

############


if st.checkbox('Show daily Graphs'):

    # Re-use the dataframe we already calculated (df_calc) which has the correct hourly logic
    # Filter only for the days we want to plot
    hourly_df_plot = df_calc.copy()
    
    # --- Generate Hourly Plots ---
    unique_dates_to_plot = hourly_df_plot['date_local'].dt.normalize().unique()

    # Initialize a BytesIO object to hold the ZIP file in memory
    zip_buffer = BytesIO()

    # Start the ZIP file context
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Your loop structure starts here
        for date in sorted(unique_dates_to_plot):
            date_normalized = pd.to_datetime(date).normalize()
            daily_hourly_data = hourly_df_plot[hourly_df_plot['date_local'].dt.normalize() == date_normalized].copy()
            daily_hourly_data = daily_hourly_data.sort_values(by='hour_local')

            # ### CHANGED: Using the pre-calculated 'is_cheap_rate' from the main logic
            daily_hourly_data['kwh_plot_cheap'] = np.where(daily_hourly_data['is_cheap_rate'], daily_hourly_data['delta_kwh'], 0)
            daily_hourly_data['kwh_plot_expensive'] = np.where(~daily_hourly_data['is_cheap_rate'], daily_hourly_data['delta_kwh'], 0)
            
            daily_hourly_data.fillna(0, inplace=True)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.set_theme(style="whitegrid")

            x_pos = daily_hourly_data['hour_local']
            width = 0.8

            ax.bar(x_pos, daily_hourly_data['kwh_plot_cheap'], color='lightgreen', width=width, edgecolor='white', label='Discount Rate')
            ax.bar(x_pos, daily_hourly_data['kwh_plot_expensive'], bottom=daily_hourly_data['kwh_plot_cheap'], color='lightcoral', width=width, edgecolor='white', label='High Rate')

            ax.set_xlabel('Hour of Day (Local Time)', fontsize=12)
            ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)

            ax.set_title(f'Hourly Energy Consumption for {date_normalized.strftime("%Y-%m-%d")}', fontsize=14)
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
            
            # Display the plot in the app (required for viewing)
            st.pyplot(fig)
            
            # --- Save Figure to Memory for Zipping ---
            
            # 1. Create a buffer for the single plot's PNG data
            plot_buffer = BytesIO()
            # 2. Save the figure to the plot buffer
            fig.savefig(plot_buffer, format="png")
            plot_buffer.seek(0) # Rewind the buffer
            
            # 3. Define the filename for this specific plot inside the ZIP
            filename = f'hourly_consumption_{date_normalized.strftime("%Y-%m-%d")}.png'
            
            # 4. Add the plot's data to the ZIP file
            zip_file.writestr(filename, plot_buffer.read())
            
            # Close the figure to free up memory
            plt.close(fig)

    # --- Create the Single Download Button OUTSIDE the Loop ---
    st.markdown("---")
    st.download_button(
        label="⬇️ Download All Plots as ZIP",
        data=zip_buffer.getvalue(), # Get the byte content of the entire ZIP file
        file_name=f'all_hourly_consumption_from_{START_DATE}_to_{END_DATE}.zip',
        mime="application/zip"
    )

# Close the initial connection
if 'conn' in locals() and conn:
    conn.close()
