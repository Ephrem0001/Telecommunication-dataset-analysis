import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage': missing_percentage
    })
    return missing_summary



def drop_columns_with_missing_values(df, missing_threshold=0.5):
    threshold = len(df) * missing_threshold
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    print(f"Columns dropped: {df.shape[1] - df_cleaned.shape[1]}")
    return df_cleaned


def impute_numerical_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
    
    numerical_cols = df.select_dtypes(include=['float64']).columns
    cols_to_impute = [col for col in numerical_cols if col not in exclude_cols]

    df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].mean())
    print(f"Imputed missing values in the following columns: {cols_to_impute}")
    return df


def plot_top_10_handsets(top_10_handsets):

    top_10_handsets.plot(kind='bar', color='skyblue')

    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Handset Type')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_3_manufacturers(df):
    top_manufacturers = df.groupby('Handset Manufacturer')['Handset Type'].count().reset_index()
    top_manufacturers = top_manufacturers.sort_values(by='Handset Type', ascending=False)
    top_3_manufacturers = top_manufacturers.head(3)

    manufacturers = top_3_manufacturers['Handset Manufacturer']
    handset_counts = top_3_manufacturers['Handset Type']

    # Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(manufacturers, handset_counts, color=['blue', 'green', 'red'])
    plt.title('Top 3 Handset Manufacturers by Number of Handsets')
    plt.xlabel('Handset Manufacturer')
    plt.ylabel('Number of Handsets')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(handset_counts, labels=manufacturers, autopct='%1.1f%%', colors=['blue', 'green', 'red'], startangle=140)
    plt.title('Market Share of Top 3 Handset Manufacturers')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def top_5_handsets_per_manufacturer(df):
    top_manufacturers = df.groupby('Handset Manufacturer')['Handset Type'].count().reset_index()
    top_manufacturers = top_manufacturers.sort_values(by='Handset Type', ascending=False)
    top_3_manufacturers = top_manufacturers.head(3)

    handset_counts = df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='Count')
    handset_counts = handset_counts.sort_values(by=['Handset Manufacturer', 'Count'], ascending=[True, False])
    
    top_3_handset_counts = handset_counts[handset_counts['Handset Manufacturer'].isin(top_3_manufacturers['Handset Manufacturer'].tolist())]
    top_5_handsets_per_manufacturer = top_3_handset_counts.groupby('Handset Manufacturer').head(5)

    return top_5_handsets_per_manufacturer


def detect_and_remove_outliers(df, numeric_columns, threshold=3):
    
    z_scores = stats.zscore(df[numeric_columns])
    outliers = (abs(z_scores) > threshold)
    df_cleaned = df[~outliers.any(axis=1)]
    num_outliers = df.shape[0] - df_cleaned.shape[0]
    print("Number of outliers removed:", num_outliers)

    return df_cleaned


def calculate_summary_statistics(df):
    df_cleaned = df.apply(pd.to_numeric, errors='coerce')
    mean_values = df_cleaned.mean()
    median_values = df_cleaned.median()
    std_dev_values = df_cleaned.std()
    variance_values = df_cleaned.var()

    return mean_values, median_values, std_dev_values, variance_values


def plot_variable_distributions(df_cleaned, variables, figsize=(24, 12)):
    num_variables = len(variables)
    num_rows = int(np.ceil(num_variables / 9)) 
    num_cols = min(9, num_variables) 

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

    flat_axes = axes.flatten() if num_rows > 1 else axes
    for i, var in enumerate(variables):
        sns.histplot(df_cleaned[var], kde=True, ax=flat_axes[i])
        flat_axes[i].set_title(var, fontsize=10)
        flat_axes[i].set_xlabel(var)  
    plt.tight_layout()
    plt.show()


def plot_application_usage(df_cleaned, applications, figsize=(15, 15)):

    if 'Total DL + UL (Bytes)' not in df_cleaned.columns:
        df_cleaned['Total DL + UL (Bytes)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']

    num_applications = len(applications)
    num_rows = int(np.ceil(num_applications / 3))  
    num_cols = min(3, num_applications)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    flat_axes = axes.flatten() if num_rows > 1 else axes

    for i, app in enumerate(applications):
        flat_axes[i].scatter(df_cleaned[app], df_cleaned['Total DL + UL (Bytes)'], alpha=0.5)
        flat_axes[i].set_title(f'{app} vs Total Data')
        flat_axes[i].set_xlabel(app)
        flat_axes[i].set_ylabel('Total DL + UL (Bytes)')
    plt.tight_layout()
    plt.show()


def analyze_application_correlations(df_cleaned, applications):

    if 'Total DL + UL (Bytes)' not in df_cleaned.columns:
        df_cleaned['Total DL + UL (Bytes)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']
        
    correlation_data = df_cleaned[applications + ['Total DL + UL (Bytes)']]
    correlation_matrix = correlation_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlation Between Applications and Total Data Usage')
    plt.show()
    

def analyze_correlation_matrix(df_cleaned, columns):
    selected_data = df_cleaned[columns]
    correlation_matrix = selected_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix Between {", ".join(columns)}')  # Dynamic title
    plt.show()
    

def perform_pca(df_cleaned, columns, n_components=2):
    selected_data = df_cleaned[columns]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)])

    explained_variance = pca.explained_variance_ratio_
    print(f'Explained Variance:')
    for i, ratio in enumerate(explained_variance):
        print(f'  - PC{i+1}: {ratio:.2f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
    plt.title('PCA Result - Projection on First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()