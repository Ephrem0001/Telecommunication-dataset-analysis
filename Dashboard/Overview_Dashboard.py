import streamlit as st  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def aggregate_customer_data(df):
    customer_aggregates = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'
    }).reset_index()

    customer_aggregates.rename(columns={
        'Avg RTT DL (ms)': 'Avg RTT DL',
        'Avg RTT UL (ms)': 'Avg RTT UL',
        'Avg Bearer TP DL (kbps)': 'Avg Throughput DL',
        'Avg Bearer TP UL (kbps)': 'Avg Throughput UL',
    }, inplace=True)

    customer_aggregates['Avg RTT'] = (customer_aggregates['Avg RTT DL'] + customer_aggregates['Avg RTT UL']) / 2
    customer_aggregates['Avg Throughput'] = (customer_aggregates['Avg Throughput DL'] + customer_aggregates['Avg Throughput UL']) / 2

    return customer_aggregates[['MSISDN/Number', 'Avg RTT', 'Avg Throughput', 'Handset Type']]

def visualize_tcp_retransmission(df):
    st.subheader('TCP Retransmission Analysis')

    top_tcp = df.nlargest(10, 'DL TP < 50 Kbps (%)')
    bottom_tcp = df.nsmallest(10, 'DL TP < 50 Kbps (%)')
    most_frequent_tcp = df['DL TP < 50 Kbps (%)'].value_counts().head(10)

    st.write("Top 10 TCP Retransmissions (DL TP < 50 Kbps)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='DL TP < 50 Kbps (%)', y='MSISDN/Number', data=top_tcp, palette='Blues_d', ax=ax)
    ax.set_title('Top 10 TCP Retransmissions (DL TP < 50 Kbps)')
    ax.set_xlabel('DL TP < 50 Kbps (%)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Bottom 10 TCP Retransmissions (DL TP < 50 Kbps)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='DL TP < 50 Kbps (%)', y='MSISDN/Number', data=bottom_tcp, palette='Reds_d', ax=ax)
    ax.set_title('Bottom 10 TCP Retransmissions (DL TP < 50 Kbps)')
    ax.set_xlabel('DL TP < 50 Kbps (%)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Most Frequent TCP Retransmissions (DL TP < 50 Kbps)")
    fig, ax = plt.subplots(figsize=(10, 6))
    most_frequent_tcp.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Most Frequent TCP Retransmissions (DL TP < 50 Kbps)')
    ax.set_xlabel('DL TP < 50 Kbps (%)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def visualize_rtt_and_throughput(df):
    st.subheader('RTT and Throughput Analysis')

    top_rtt = df.nlargest(10, 'Avg RTT DL (ms)')
    bottom_rtt = df.nsmallest(10, 'Avg RTT DL (ms)')
    most_frequent_rtt = df['Avg RTT DL (ms)'].value_counts().head(10)

    st.write("Top 10 RTT (Avg RTT DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Avg RTT DL (ms)', y='MSISDN/Number', data=top_rtt, palette='Greens_d', ax=ax)
    ax.set_title('Top 10 RTT (Avg RTT DL)')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Bottom 10 RTT (Avg RTT DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Avg RTT DL (ms)', y='MSISDN/Number', data=bottom_rtt, palette='Oranges_d', ax=ax)
    ax.set_title('Bottom 10 RTT (Avg RTT DL)')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Most Frequent RTT (Avg RTT DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    most_frequent_rtt.plot(kind='bar', color='cyan', ax=ax)
    ax.set_title('Most Frequent RTT (Avg RTT DL)')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    top_throughput = df.nlargest(10, 'Avg Bearer TP DL (kbps)')
    bottom_throughput = df.nsmallest(10, 'Avg Bearer TP DL (kbps)')
    most_frequent_throughput = df['Avg Bearer TP DL (kbps)'].value_counts().head(10)

    st.write("Top 10 Throughput (Avg Bearer TP DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=top_throughput, palette='Blues_d', ax=ax)
    ax.set_title('Top 10 Throughput (Avg Bearer TP DL)')
    ax.set_xlabel('Avg Bearer TP DL (kbps)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Bottom 10 Throughput (Avg Bearer TP DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=bottom_throughput, palette='Reds_d', ax=ax)
    ax.set_title('Bottom 10 Throughput (Avg Bearer TP DL)')
    ax.set_xlabel('Avg Bearer TP DL (kbps)')
    ax.set_ylabel('MSISDN/Number')
    st.pyplot(fig)

    st.write("Most Frequent Throughput (Avg Bearer TP DL)")
    fig, ax = plt.subplots(figsize=(10, 6))
    most_frequent_throughput.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Most Frequent Throughput (Avg Bearer TP DL)')
    ax.set_xlabel('Avg Bearer TP DL (kbps)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def visualize_throughput_and_tcp_per_handset(df):
    st.subheader('Throughput and TCP Analysis by Handset Type')

    st.write("Average Throughput per Handset Type")
    throughput_per_handset = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    throughput_per_handset.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Throughput per Handset Type')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Average Throughput (kbps)')
    ax.set_xticklabels(throughput_per_handset.index, rotation=45)
    st.pyplot(fig)

    st.write("Average TCP Retransmission per Handset Type")
    tcp_per_handset = df.groupby('Handset Type')['DL TP < 50 Kbps (%)'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    tcp_per_handset.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_title('Average TCP Retransmission per Handset Type')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Average TCP Retransmission (%)')
    ax.set_xticklabels(tcp_per_handset.index, rotation=45)
    st.pyplot(fig)

def process_and_cluster_customers(df, n_clusters=3):
    customer_aggregates = aggregate_customer_data(df)

    experience_metrics = customer_aggregates[['Avg RTT', 'Avg Throughput']]
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(experience_metrics)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    customer_aggregates['Cluster'] = kmeans.fit_predict(scaled_metrics)

    for cluster in customer_aggregates['Cluster'].unique():
        st.write(f'Cluster {cluster}:')
        st.write(customer_aggregates[customer_aggregates['Cluster'] == cluster].describe())
    
    return customer_aggregates

def fetch_data(file_path):
    return pd.read_csv(file_path)

def calculate_missing_percentage(df):
    return df.isnull().mean() * 100

def drop_columns_with_missing_values(df, threshold=50):
    return df.dropna(axis=1, thresh=len(df) * (1 - threshold / 100))

def impute_numerical_columns(df):
    return df.fillna(df.median())

def plot_top_10_handsets(df):
    st.subheader('Top 10 Handsets by Average Throughput')

    top_handsets = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_handsets.index, y=top_handsets.values, palette='viridis', ax=ax)
    ax.set_title('Top 10 Handsets by Average Throughput')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Average Throughput (kbps)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main(uploaded_file):

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Overview")
        st.write(df.head())

        if st.checkbox('Show missing data percentage'):
            missing_percentage = calculate_missing_percentage(df)
            st.write("Missing Data Percentage:")
            st.write(missing_percentage)

        if st.checkbox('Drop columns with missing values'):
            df = drop_columns_with_missing_values(df)
            st.write("Updated Data")
            st.write(df.head())

        if st.checkbox('Impute missing values'):
            df = impute_numerical_columns(df)
            st.write("Updated Data")
            st.write(df.head())

        if st.checkbox('Visualize TCP Retransmission'):
            visualize_tcp_retransmission(df)

        if st.checkbox('Visualize RTT and Throughput'):
            visualize_rtt_and_throughput(df)

        if st.checkbox('Visualize Throughput and TCP by Handset Type'):
            visualize_throughput_and_tcp_per_handset(df)

        if st.checkbox('Perform Clustering'):
            customer_data = process_and_cluster_customers(df)
            st.write("Customer Clustering Results:")
            st.write(customer_data)

        if st.checkbox('Show Top 10 Handsets by Throughput'):
            plot_top_10_handsets(df)


def fetch_data(file_path):
    df = pd.read_csv(file_path)
    return df


def calculate_missing_percentage(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing Values': missing_count, 'Percentage': missing_percentage})
    return missing_summary


def drop_columns_with_missing_values(df, missing_threshold=0.5):
    threshold = len(df) * missing_threshold
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    st.write(f"Columns dropped: {df.shape[1] - df_cleaned.shape[1]}")
    return df_cleaned


def impute_numerical_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
    
    numerical_cols = df.select_dtypes(include=['float64']).columns
    cols_to_impute = [col for col in numerical_cols if col not in exclude_cols]
    df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].mean())
    st.write(f"Imputed missing values in the following columns: {cols_to_impute}")
    return df


def plot_top_10_handsets(top_10_handsets):
    fig, ax = plt.subplots()
    top_10_handsets.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Top 10 Handsets Used by Customers')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Number of Users')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def visualize_and_cluster_data(df):
    experience_metrics = df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(experience_metrics)
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scaled_metrics)
    for cluster in df['Cluster'].unique():
        st.write(f'Cluster {cluster}:')
        st.write(df[df['Cluster'] == cluster].describe())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['Avg RTT DL (ms)'], y=df['Avg Bearer TP DL (kbps)'], hue=df['Cluster'], palette='viridis', ax=ax)
    ax.set_title('K-Means Clustering Results')
    ax.set_xlabel('Avg RTT DL (ms)')
    ax.set_ylabel('Avg Bearer TP DL (kbps)')
    st.pyplot(fig)


st.title("User Overview Dashboard")
uploaded_file = st.file_uploader("https://docs.google.com/spreadsheets/d/e/2PACX-1vRRhL-TLOPGWO7T450wnFpXzmX0_rQPBg1eIcggit1XGpc4dWekJz1B00xUYaUIjGH-FSVTSsdupALv/pub?output=csv", type="csv")

if uploaded_file:
    data = fetch_data(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    missing_summary = calculate_missing_percentage(data)
    st.write("Missing Data Summary:")
    st.write(missing_summary)

    data_cleaned = drop_columns_with_missing_values(data)
    data_imputed = impute_numerical_columns(data_cleaned)
    top_10_handsets = data['Handset Type'].value_counts().head(10)
    plot_top_10_handsets(top_10_handsets)
    visualize_and_cluster_data(data_imputed)
else:
    st.write("Please upload a CSV file to proceed.")


st.title("User_Experience Dashboard")

st.sidebar.header("Dashboard")
uploaded_files = st.file_uploader("https://docs.google.com/spreadsheets/d/e/2PACX-1vTrZZ1sBTwmSH69C6KLzTPpAsoeRPwNuX-9mbhNKsVLuw1zKNweWfizBMSOFLmx8S17aLxnFi0Fg3KT/pub?output=csv", type="csv")
main(uploaded_files)
