import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Non-Farm Payrolls OLAP Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Database Configuration ---
# Replace with your PostgreSQL credentials
DB_HOST = "localhost"
DB_NAME = "ETL"
DB_USER = "postgres"
DB_PASSWORD = "Admin@0416"
DB_PORT = "5432"


# --- CUSTOM CSS ---
def local_css():
    """Injects custom CSS for styling the app."""
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .main-header {
            text-align: center;
            font-weight: bold;
            color: #1E3A8A; /* A deep blue color */
        }
        .stDataFrame, .stPlotlyChart {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
            padding: 15px;
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# --- DATABASE & DATA LOADING ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data() -> pd.DataFrame:
    """
    Connects to the PostgreSQL database and fetches nonfarm payrolls data.
    Returns a pandas DataFrame.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        # Load data, ensuring it's sorted by date for time series analysis
        query = "SELECT * FROM nonfarm_payrolls ORDER BY date ASC;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert 'date' column to datetime objects and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except psycopg2.OperationalError as e:
        st.error(f"❌ Database Connection Failed: {e}")
        st.error("Please ensure your PostgreSQL server is running and credentials are correct.")
        return None
    except Exception as e:
        # Handle cases where the table might not exist
        if "relation \"nonfarm_payrolls\" does not exist" in str(e):
            st.error("❌ Database Error: The table 'nonfarm_payrolls' was not found.")
            st.info("Please run the ETL script first to create and populate the table.")
        else:
            st.error(f"❌ An unexpected error occurred: {e}")
        return None

# --- OLAP ANALYSIS FUNCTIONS ---

def perform_slicing(df):
    """Displays Slicing analysis: focusing on specific time subsets."""
    st.header("🔪 Slicing: Analyzing Specific Time Periods")

    # --- Question 1 ---
    st.subheader("1. Average Total Payroll Employment by Year")
    
    # Add a slider to select the year range
    min_year = int(df.index.year.min())
    max_year = int(df.index.year.max())
    
    selected_years = st.slider(
        "Select the year range to analyze:",
        min_value=min_year,
        max_value=max_year,
        value=(2010, max_year) # A sensible default
    )
    start_year, end_year = selected_years

    # Filter the dataframe based on the slider
    slice_df_q1 = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
    avg_by_year = slice_df_q1['total_nonfarm'].resample('Y').mean() / 1000 # In thousands
    avg_by_year.index = avg_by_year.index.year
    
    fig = px.bar(
        avg_by_year,
        y="total_nonfarm",
        title=f"Average Monthly Non-Farm Employment ({start_year}–{end_year})",
        labels={'value': 'Average Employment (in thousands)', 'date': 'Year'},
        text=avg_by_year.apply(lambda x: f'{x:,.0f}K')
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Conceptual SQL Logic"):
        st.code(f"""
SELECT
    EXTRACT(YEAR FROM date) AS year,
    AVG(total_nonfarm) AS average_employment
FROM
    nonfarm_payrolls
WHERE
    EXTRACT(YEAR FROM date) BETWEEN {start_year} AND {end_year}
GROUP BY
    year
ORDER BY
    year;
        """, language="sql")

    # --- Question 2 ---
    st.subheader("2. 2020 vs. 2019: A Tale of Two Years (March–December)")
    slice_df_q2 = df[df.index.year.isin([2019, 2020]) & (df.index.month >= 3)].copy()
    slice_df_q2['year'] = slice_df_q2.index.year
    slice_df_q2['month'] = slice_df_q2.index.month_name()
    
    st.write("Comparing monthly employment levels during the pandemic onset in 2020 against the baseline of 2019.")

    fig = px.line(
        slice_df_q2,
        x='month',
        y='total_nonfarm',
        color='year',
        title="Monthly Employment: 2020 vs. 2019 (Mar-Dec)",
        labels={'total_nonfarm': 'Total Employment (in thousands)', 'month': 'Month'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Conceptual SQL Logic"):
        st.code("""
SELECT
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    total_nonfarm
FROM
    nonfarm_payrolls
WHERE
    EXTRACT(YEAR FROM date) IN (2019, 2020)
    AND EXTRACT(MONTH FROM date) >= 3
ORDER BY
    year, month;
        """, language="sql")


def perform_dicing(df):
    """Displays Dicing analysis: filtering based on multiple criteria."""
    st.header("🎲 Dicing: Filtering on Multiple Conditions")

    # --- Question 1 ---
    st.subheader("1. Major Employment Drops (>2%) and Recovery Time")
    
    diced_df_q1 = df[df['mom_change_pct'] < -2.0].copy()
    st.write("Identifying months with a significant month-over-month percentage drop and calculating the time to recover to the previous peak employment level.")

    if not diced_df_q1.empty:
        recovery_data = []
        for drop_date, row in diced_df_q1.iterrows():
            prior_peak_date = drop_date - pd.DateOffset(months=1)
            prior_peak_value = df.loc[prior_peak_date, 'total_nonfarm']
            
            recovery_df = df[df.index > drop_date]
            recovery_point = recovery_df[recovery_df['total_nonfarm'] >= prior_peak_value].first_valid_index()
            
            if recovery_point:
                time_to_recover = (recovery_point.year - drop_date.year) * 12 + (recovery_point.month - drop_date.month)
                recovery_data.append({
                    "Drop Date": drop_date.strftime('%Y-%m'),
                    "Prior Peak Employment": f"{prior_peak_value:,}K",
                    "Drop Percentage": f"{row['mom_change_pct']:.2f}%",
                    "Recovery Date": recovery_point.strftime('%Y-%m'),
                    "Months to Recover": time_to_recover
                })
        
        st.table(pd.DataFrame(recovery_data))
    else:
        st.info("No months found with a drop greater than 2%.")

    with st.expander("Show Logic Explanation (Procedural)"):
        st.markdown("""
        This analysis is more procedural than a single SQL query:
        1. **Filter**: Identify all months where `mom_change_pct < -2.0`.
        2. **For each drop**:
           - Get the employment level from the month just before the drop (the 'prior peak').
           - Search in the data *after* the drop date.
           - Find the first month where employment is greater than or equal to the 'prior peak' value.
           - Calculate the difference in months between the drop and recovery dates.
        """)

    # --- Question 2 ---
    st.subheader("2. Highest Payroll Growth Month within Q4")
    diced_df_q2 = df[df.index.quarter == 4].copy()
    diced_df_q2['month'] = diced_df_q2.index.month_name()
    avg_growth_q4 = diced_df_q2.groupby('month')['mom_change_abs'].mean().sort_values(ascending=False)
    
    st.write("Analyzing the final quarter of each year to find which month (Oct, Nov, or Dec) consistently shows the highest average gain in employment.")
    st.metric(
        label="Best Performing Month in Q4",
        value=avg_growth_q4.index[0],
        delta=f"{avg_growth_q4.iloc[0]:,.0f}K avg. gain"
    )

    fig = px.bar(
        avg_growth_q4,
        title="Average Month-over-Month Payroll Gain in Q4",
        labels={'value': 'Average Absolute Gain (in thousands)', 'index': 'Month'},
        text=avg_growth_q4.apply(lambda x: f'{x:,.0f}K')
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Conceptual SQL Logic"):
        st.code("""
SELECT
    EXTRACT(MONTH FROM date) AS month,
    AVG(mom_change_abs) AS average_absolute_gain
FROM
    nonfarm_payrolls
WHERE
    EXTRACT(QUARTER FROM date) = 4
GROUP BY
    month
ORDER BY
    average_absolute_gain DESC;
        """, language="sql")


def perform_rollup(df):
    """Displays Roll-up analysis: aggregating data to higher levels."""
    st.header("⬆️ Roll-up: Aggregating Data")

    # --- Question 1 ---
    st.subheader("1. Quarterly and Yearly Employment Growth Rates")

    # Yearly Roll-up
    yearly_rollup = df['total_nonfarm'].resample('Y').last()
    yearly_growth = yearly_rollup.pct_change() * 100

    # Quarterly Roll-up
    quarterly_rollup = df['total_nonfarm'].resample('Q').last()
    quarterly_growth = quarterly_rollup.pct_change() * 100

    st.write("Aggregating monthly data to see the bigger picture of quarterly and annual growth trends.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Year-over-Year Growth (%)")
        st.dataframe(yearly_growth.to_frame(name="YoY Growth (%)").style.format("{:.2f}%"))
    with col2:
        st.write("Quarter-over-Quarter Growth (%)")
        st.dataframe(quarterly_growth.to_frame(name="QoQ Growth (%)").style.format("{:.2f}%"))

    fig = px.bar(
        yearly_growth.tail(20), # Show last 20 years for clarity
        title="Year-over-Year Employment Growth Rate",
        labels={'value': 'Growth Rate (%)', 'date': 'Year'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Show Conceptual SQL Logic (for Yearly Aggregation)"):
        st.code("""
-- Step 1: Get the last value for each year
WITH YearlyData AS (
    SELECT
        DATE_TRUNC('year', date) as year_start,
        (SELECT total_nonfarm FROM nonfarm_payrolls
         WHERE DATE_TRUNC('year', date) = T.year_start
         ORDER BY date DESC LIMIT 1) as end_of_year_employment
    FROM nonfarm_payrolls T
    GROUP BY year_start
)
-- Step 2: Calculate year-over-year change (using window function)
SELECT
    year_start,
    end_of_year_employment,
    (end_of_year_employment - LAG(end_of_year_employment, 1) OVER (ORDER BY year_start))
    / LAG(end_of_year_employment, 1) OVER (ORDER BY year_start) * 100.0 AS yoy_growth_pct
FROM YearlyData;
        """, language="sql")
        
    # --- Question 2 ---
    st.subheader("2. Average Employment: 2010s vs. 2000s")
    avg_2000s = df[(df.index.year >= 2000) & (df.index.year <= 2009)]['total_nonfarm'].mean()
    avg_2010s = df[(df.index.year >= 2010) & (df.index.year <= 2019)]['total_nonfarm'].mean()
    
    st.write("Comparing the average monthly employment across two different decades.")
    
    col1, col2 = st.columns(2)
    col1.metric("Avg. Employment in 2000s", f"{avg_2000s:,.0f}K")
    col2.metric("Avg. Employment in 2010s", f"{avg_2010s:,.0f}K", f"{((avg_2010s - avg_2000s)/avg_2000s)*100:.2f}% vs 2000s")
    
    with st.expander("Show Conceptual SQL Logic"):
        st.code("""
SELECT
    CASE
        WHEN EXTRACT(YEAR FROM date) BETWEEN 2000 AND 2009 THEN '2000s'
        WHEN EXTRACT(YEAR FROM date) BETWEEN 2010 AND 2019 THEN '2010s'
    END AS decade,
    AVG(total_nonfarm) as average_employment
FROM
    nonfarm_payrolls
WHERE
    EXTRACT(YEAR FROM date) BETWEEN 2000 AND 2019
GROUP BY
    decade;
        """, language="sql")


def perform_drilldown(df):
    """Displays Drill-Down analysis: moving from summary to detail."""
    st.header("⬇️ Drill-Down: From Summary to Detail")

    # --- Question 1 ---
    st.subheader("1. Monthly Contributions in the Year with Highest Gain")
    
    yearly_change = df['total_nonfarm'].resample('Y').last().diff()
    best_year = yearly_change.idxmax().year
    
    st.write(f"The year with the highest absolute employment gain was **{best_year}**, with a total increase of **{yearly_change.max():,.0f}K**. Here's the monthly breakdown of that gain.")
    
    best_year_df = df[df.index.year == best_year].copy()
    
    fig = px.bar(
        best_year_df,
        y='mom_change_abs',
        title=f"Monthly Employment Gains in {best_year}",
        labels={'mom_change_abs': 'Month-over-Month Gain (in thousands)', 'date': 'Month'},
        text=best_year_df['mom_change_abs'].apply(lambda x: f'{x:,.0f}K')
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Monthly Gain (in thousands)")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Show Conceptual SQL Logic"):
        st.markdown(f"""
        This is a two-step process:
        1. **First, find the year with the highest gain.** This is complex in pure SQL but involves comparing the end-of-year values.
        2. **Then, drill down into that year's data.** Once the best year (`{best_year}`) is identified, the query is straightforward:
        """)
        st.code(f"""
SELECT
    date,
    mom_change_abs
FROM
    nonfarm_payrolls
WHERE
    EXTRACT(YEAR FROM date) = {best_year}
ORDER BY
    date;
        """, language="sql")
        
    # --- Question 2 ---
    st.subheader("2. Analysis of the Sharpest Single-Month Drop")
    
    sharpest_drop_date = df['mom_change_abs'].idxmin()
    sharpest_drop_row = df.loc[sharpest_drop_date]
    
    st.write("Drilling down to the single worst month on record for employment change.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Month of Sharpest Drop", sharpest_drop_date.strftime('%B %Y'))
    col2.metric("Absolute Drop", f"{sharpest_drop_row['mom_change_abs']:,.0f}K")
    col3.metric("Percentage Drop", f"{sharpest_drop_row['mom_change_pct']:.2f}%")

    st.info("💡 **Further Drill-Down:** A common next step would be to drill down into weekly or daily data. However, this is not possible as the **FRED 'PAYEMS' series is a monthly dataset**. A more granular analysis would require a different data source.")

# --- MAIN APP LAYOUT ---
def main():
    """Main function to run the Streamlit app."""
    local_css()
    st.markdown("<h1 class='main-header'>U.S. Non-Farm Payrolls OLAP Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("An interactive tool to analyze employment trends using OLAP operations on PostgreSQL data.")
    
    # Load data and handle potential errors
    data = load_data()
    
    if data is not None:
        st.success("✅ Database connection successful and data loaded.")
        
        # Sidebar for navigation
        st.sidebar.title("📊 OLAP Operations")
        analysis_type = st.sidebar.radio(
            "Choose an analysis type:",
            ("Slicing", "Dicing", "Roll-up", "Drill-Down")
        )
        
        # Display selected analysis
        if analysis_type == "Slicing":
            perform_slicing(data)
        elif analysis_type == "Dicing":
            perform_dicing(data)
        elif analysis_type == "Roll-up":
            perform_rollup(data)
        elif analysis_type == "Drill-Down":
            perform_drilldown(data)
    else:
        st.warning("Data could not be loaded. Please check the error messages above.")

if __name__ == "__main__":
    main()

