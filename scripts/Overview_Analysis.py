import os
import psycopg2
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



def drop_columns_with_missing_values(df, missing_threshold=0.5):
    """
    Drop columns in a DataFrame that have more than a certain percentage of missing values.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - missing_threshold (float): The threshold for dropping columns. Default is 0.5 (50%).
    
    Returns:
    - pd.DataFrame: A DataFrame with columns dropped based on the missing value threshold.
    """
    # Calculate the threshold number of non-missing values per column
    threshold = len(df) * missing_threshold
    
    # Drop columns that have more than the threshold of missing values
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    
    print(f"Columns dropped: {df.shape[1] - df_cleaned.shape[1]}")
    return df_cleaned

# Example usage:
# Assuming you have a DataFrame `df`
# cleaned_df = drop_columns_with_missing_values(df)



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
    # Grouping by Handset Manufacturer and counting the number of handsets
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
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
    plt.tight_layout()
    plt.show()


def top_5_handsets_per_manufacturer(df):
    """
    Finds the top 5 handset types for the top 3 handset manufacturers in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing handset data.

    Returns:
        pandas.DataFrame: A DataFrame with the top 5 handset types for each of the top 3 manufacturers.
    """

    # Group by manufacturer and count handset types
    top_manufacturers = df.groupby('Handset Manufacturer')['Handset Type'].count().reset_index()

    # Sort manufacturers by handset count descending
    top_manufacturers = top_manufacturers.sort_values(by='Handset Type', ascending=False)

    # Get the top 3 manufacturers
    top_3_manufacturers = top_manufacturers.head(3)

    # Group by manufacturer and handset type, count occurrences
    handset_counts = df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='Count')

    # Sort handset counts by manufacturer ascending and count descending
    handset_counts = handset_counts.sort_values(by=['Handset Manufacturer', 'Count'], ascending=[True, False])

    # Filter handset counts for the top 3 manufacturers
    top_3_handset_counts = handset_counts[handset_counts['Handset Manufacturer'].isin(top_3_manufacturers['Handset Manufacturer'].tolist())]

    # Get the top 5 handsets for each manufacturer
    top_5_handsets_per_manufacturer = top_3_handset_counts.groupby('Handset Manufacturer').head(5)

    return top_5_handsets_per_manufacturer


def detect_and_remove_outliers(df, numeric_columns, threshold=3):
    """
    Detects and removes outliers from a DataFrame using Z-score method.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numeric data.
        numeric_columns (list): A list of column names to consider for outlier detection.
        threshold (float): The Z-score threshold for identifying outliers.

    Returns:
        pandas.DataFrame: A DataFrame with outliers removed.
    """

    # Calculate Z-scores for numeric columns
    z_scores = stats.zscore(df[numeric_columns])

    # Identify outliers based on threshold
    outliers = (abs(z_scores) > threshold)

    # Remove rows with outliers
    df_cleaned = df[~outliers.any(axis=1)]

    # Report the number of outliers removed
    num_outliers = df.shape[0] - df_cleaned.shape[0]
    print("Number of outliers removed:", num_outliers)

    return df_cleaned


def calculate_summary_statistics(df):
    # Convert all columns to numeric, coercing errors
    df_cleaned = df.apply(pd.to_numeric, errors='coerce')

    # Calculate summary statistics
    mean_values = df_cleaned.mean()
    median_values = df_cleaned.median()
    std_dev_values = df_cleaned.std()
    variance_values = df_cleaned.var()

    return mean_values, median_values, std_dev_values, variance_values


def plot_variable_distributions(df_cleaned, variables, figsize=(24, 12)):
  # Calculate the number of rows and columns based on the number of variables
  num_variables = len(variables)
  num_rows = int(np.ceil(num_variables / 9))  # Use ceil to handle uneven distribution
  num_cols = min(9, num_variables)  # Limit to 9 columns for readability

  # Create the figure and axes
  fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

  # Flatten the axes for easy iteration (if necessary)
  flat_axes = axes.flatten() if num_rows > 1 else axes

  # Loop through variables and plot the corresponding graph
  for i, var in enumerate(variables):
    sns.histplot(df_cleaned[var], kde=True, ax=flat_axes[i])
    flat_axes[i].set_title(var, fontsize=10)
    flat_axes[i].set_xlabel(var)  # Add x-axis label

  # Adjust layout and display the plot
  plt.tight_layout()
  plt.show()


def plot_application_usage(df_cleaned, applications, figsize=(15, 15)):
  """
  Plots scatter plots for application usage against total data usage.

  Args:
      df_cleaned (pd.DataFrame): The DataFrame containing cleaned data.
      applications (list): A list of application names to plot.
      figsize (tuple, optional): The figure size for the plot. Defaults to (15, 15).
  """

  # Create a new column for total data usage (if not already present)
  if 'Total DL + UL (Bytes)' not in df_cleaned.columns:
    df_cleaned['Total DL + UL (Bytes)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']

  # Calculate the number of rows and columns based on the number of applications
  num_applications = len(applications)
  num_rows = int(np.ceil(num_applications / 3))  # Use ceil to handle uneven distribution
  num_cols = min(3, num_applications)  # Limit to 3 columns for readability

  # Create the figure and axes
  fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

  # Flatten the axes for easy iteration (if necessary)
  flat_axes = axes.flatten() if num_rows > 1 else axes

  # Loop through applications and plot the corresponding scatter plot
  for i, app in enumerate(applications):
    flat_axes[i].scatter(df_cleaned[app], df_cleaned['Total DL + UL (Bytes)'], alpha=0.5)
    flat_axes[i].set_title(f'{app} vs Total Data')
    flat_axes[i].set_xlabel(app)
    flat_axes[i].set_ylabel('Total DL + UL (Bytes)')

  # Adjust layout and display the plot
  plt.tight_layout()
  plt.show()

# Example usage
# Assuming you have a DataFrame named 'df_cleaned' and the list 'applications' defined


def analyze_application_correlations(df_cleaned, applications):
  """
  Analyzes correlations between applications and total data usage.

  Args:
      df_cleaned (pd.DataFrame): The DataFrame containing cleaned data.
      applications (list): A list of application names.
  """

  # Create a new column for total data usage (if not already present)
  if 'Total DL + UL (Bytes)' not in df_cleaned.columns:
    df_cleaned['Total DL + UL (Bytes)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']

  # Select application columns and total data usage
  correlation_data = df_cleaned[applications + ['Total DL + UL (Bytes)']]

  # Compute correlation matrix
  correlation_matrix = correlation_data.corr()

  # Plot the heatmap to visualize all correlations
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
  plt.title('Correlation Between Applications and Total Data Usage')
  plt.show()
  
  
def analyze_correlation_matrix(df_cleaned, columns):
  """
  Analyzes and visualizes the correlation matrix for a set of columns.

  Args:
      df_cleaned (pd.DataFrame): The DataFrame containing cleaned data.
      columns (list): A list of column names to analyze.
  """

  # Select the specified columns
  selected_data = df_cleaned[columns]

  # Compute correlation matrix
  correlation_matrix = selected_data.corr()

  # Plot the correlation matrix as a heatmap
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
  plt.title(f'Correlation Matrix Between {", ".join(columns)}')  # Dynamic title
  plt.show()
  

def perform_pca(df_cleaned, columns, n_components=2):
  """
  Performs PCA analysis and visualization on a subset of columns.

  Args:
      df_cleaned (pd.DataFrame): The DataFrame containing cleaned data.
      columns (list): A list of column names to analyze using PCA.
      n_components (int, optional): The number of principal components to compute. Defaults to 2.
  """

  # Select the specified columns
  selected_data = df_cleaned[columns]

  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(selected_data)

  # Apply PCA with the specified number of components
  pca = PCA(n_components=n_components)
  pca_data = pca.fit_transform(scaled_data)

  # Create a DataFrame with the PCA results
  pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)])

  # Print the explained variance ratios
  explained_variance = pca.explained_variance_ratio_
  print(f'Explained Variance:')
  for i, ratio in enumerate(explained_variance):
    print(f'  - PC{i+1}: {ratio:.2f}')

  # Plot the PCA results
  plt.figure(figsize=(8, 6))
  plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
  plt.title('PCA Result - Projection on First Two Principal Components')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.grid(True)
  plt.show()