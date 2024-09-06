import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv() # Load environment variables from .env file

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def load_data_using_sqlalchemy(query):

    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_missing_percentage(df):
    # Calculate the number of missing values in each column
    missing_count = df.isnull().sum()
    
    # Calculate the percentage of missing values
    missing_percentage = (missing_count / len(df)) * 100
    
    # Create a DataFrame to display the results
    missing_summary = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage': missing_percentage
    })
    
    return missing_summary

