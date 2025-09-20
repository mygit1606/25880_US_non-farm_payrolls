# etl_pipeline.py

import os
import pandas as pd
import psycopg2
from fredapi import Fred
from io import StringIO
import sys

# --- CONFIGURATION ---
# It's best practice to use environment variables for credentials.
# Set them in your shell before running the script, e.g.:
# export FRED_API_KEY='your_fred_api_key'
# export DB_NAME='your_db_name'
# export DB_USER='your_db_user'
# export DB_PASSWORD='your_db_password'

# FRED_API_KEY = os.getenv('FRED_API_KEY')
# DB_NAME = os.getenv('DB_NAME', 'postgres')
# DB_USER = os.getenv('DB_USER', 'postgres')
# DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
# DB_HOST = os.getenv('DB_HOST', 'localhost')
# DB_PORT = os.getenv('DB_PORT', '5432')

# --- CONFIGURATION ---

# WARNING: Hardcoding credentials is not secure. Use for testing only.
FRED_API_KEY = 'd46feb9deca87646f74eb02613304647'
DB_NAME = 'ETL'
DB_USER = 'postgres'
DB_PASSWORD = 'Admin@0416'
DB_HOST = 'localhost' # or your db host
DB_PORT = '5432'      # or your db port

# --- 1. EXTRACT ---
def extract_fred_data(api_key: str, series_id: str = 'PAYEMS') -> pd.Series:
    """
    Extracts a specified time series from the FRED API.

    Args:
        api_key (str): Your FRED API key.
        series_id (str): The series ID to retrieve (e.g., 'PAYEMS').

    Returns:
        pd.Series: A pandas Series with the time series data, or None on failure.
    """
    if not api_key:
        print("❌ FRED API key not found. Please set the FRED_API_KEY environment variable.")
        sys.exit(1) # Exit the script if API key is missing
        
    try:
        print(f"📈 Extracting series '{series_id}' from FRED...")
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        print("✅ Extraction complete.")
        return data
    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
        return None

# --- 2. TRANSFORM ---
def transform_payroll_data(raw_data: pd.Series) -> pd.DataFrame:
    """
    Transforms raw payroll data into a clean DataFrame with month-over-month changes.

    Args:
        raw_data (pd.Series): Raw time series data from FRED.

    Returns:
        pd.DataFrame: A transformed DataFrame.
    """
    print("⚙️  Transforming data...")
    # Load into a DataFrame and name the column
    df = pd.DataFrame(raw_data, columns=['total_nonfarm'])

    # Calculate month-over-month absolute change
    df['mom_change_abs'] = df['total_nonfarm'].diff()

    # Calculate month-over-month percentage change
    df['mom_change_pct'] = df['total_nonfarm'].pct_change() * 100

    # Clean the data by dropping the first row with NaN values
    df.dropna(inplace=True)

    # Reset the index to turn the date index into a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    # Ensure data types are correct for DB loading
    df['total_nonfarm'] = df['total_nonfarm'].astype('int64')
    df['mom_change_abs'] = df['mom_change_abs'].astype('int64')
    df['mom_change_pct'] = df['mom_change_pct'].round(4)


    print("✅ Transformation complete.")
    return df

# --- 3. LOAD ---
def load_data_to_postgres(df: pd.DataFrame, db_params: dict):
    """
    Loads the transformed data into a PostgreSQL table using the efficient copy_from method.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        db_params (dict): Dictionary with database connection parameters.
    """
    conn = None
    table_name = 'nonfarm_payrolls'
    
    # SQL to create the table
    create_table_query = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        date DATE PRIMARY KEY,
        total_nonfarm BIGINT,
        mom_change_abs BIGINT,
        mom_change_pct NUMERIC(10, 4)
    );
    """

    print(f"🚚 Loading data into PostgreSQL table '{table_name}'...")
    try:
        # Establish connection
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Create the table
        cur.execute(create_table_query)
        print(f"   - Table '{table_name}' created successfully.")

        # Prepare data for bulk loading
        # Use StringIO to create an in-memory "file"
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, sep='\t')
        buffer.seek(0) # Rewind the buffer to the beginning

        # Use copy_from for an efficient bulk insert
        cur.copy_from(buffer, table_name, sep='\t', columns=df.columns)
        
        # Commit the transaction
        conn.commit()
        print(f"✅ Load complete. {len(df)} rows loaded into '{table_name}'.")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.rollback() # Rollback the transaction on error
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # Ensure the connection is always closed
        if conn:
            conn.close()
            print("   - Database connection closed.")


# --- Main ETL Execution ---
if __name__ == "__main__":
    print("🚀 Starting ETL pipeline...")
    
    # Database connection parameters from environment variables
    db_connection_params = {
        'dbname': DB_NAME,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': DB_HOST,
        'port': DB_PORT
    }

    # 1. Extract
    payems_data = extract_fred_data(api_key=FRED_API_KEY)

    if payems_data is not None and not payems_data.empty:
        # 2. Transform
        transformed_df = transform_payroll_data(payems_data)
        
        # 3. Load
        load_data_to_postgres(transformed_df, db_connection_params)
        
        print("🎉 ETL pipeline finished successfully!")
    else:
        print("ETL pipeline stopped due to extraction failure.")