import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def aggregate_customer_data(df):
    # Step 2: Aggregate per customer
    customer_aggregates = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'  # Assuming customers have one handset type
    }).reset_index()

    # Rename columns for clarity
    customer_aggregates.rename(columns={
        'Avg RTT DL (ms)': 'Avg RTT DL',
        'Avg RTT UL (ms)': 'Avg RTT UL',
        'Avg Bearer TP DL (kbps)': 'Avg Throughput DL',
        'Avg Bearer TP UL (kbps)': 'Avg Throughput UL',
    }, inplace=True)

    # Calculate average RTT and Throughput
    customer_aggregates['Avg RTT'] = (customer_aggregates['Avg RTT DL'] + customer_aggregates['Avg RTT UL']) / 2
    customer_aggregates['Avg Throughput'] = (customer_aggregates['Avg Throughput DL'] + customer_aggregates['Avg Throughput UL']) / 2

    # Final result
    customer_aggregates = customer_aggregates[['MSISDN/Number', 'Avg RTT', 'Avg Throughput', 'Handset Type']]
    
    return customer_aggregates


def visualize_tcp_retransmission(df):
    # Top 10 TCP retransmission (DL TP < 50 Kbps)
    top_tcp = df.nlargest(10, 'DL TP < 50 Kbps (%)')
    bottom_tcp = df.nsmallest(10, 'DL TP < 50 Kbps (%)')
    most_frequent_tcp = df['DL TP < 50 Kbps (%)'].value_counts().head(10)

    # Setting style for the plots
    sns.set(style="whitegrid")

    # Top 10 TCP Retransmission Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DL TP < 50 Kbps (%)', y='MSISDN/Number', data=top_tcp, palette='Blues_d')
    plt.title('Top 10 TCP Retransmissions (DL TP < 50 Kbps)')
    plt.xlabel('DL TP < 50 Kbps (%)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    # Bottom 10 TCP Retransmission Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DL TP < 50 Kbps (%)', y='MSISDN/Number', data=bottom_tcp, palette='Reds_d')
    plt.title('Bottom 10 TCP Retransmissions (DL TP < 50 Kbps)')
    plt.xlabel('DL TP < 50 Kbps (%)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    # Most Frequent TCP Retransmissions Visualizations
    plt.figure(figsize=(10, 6))
    most_frequent_tcp.plot(kind='bar', color='purple')
    plt.title('Most Frequent TCP Retransmissions (DL TP < 50 Kbps)')
    plt.xlabel('DL TP < 50 Kbps (%)')
    plt.ylabel('Frequency')
    plt.show()


def visualize_rtt_and_throughput(df):
    # Top 10 RTT (Avg RTT DL)
    top_rtt = df.nlargest(10, 'Avg RTT DL (ms)')
    bottom_rtt = df.nsmallest(10, 'Avg RTT DL (ms)')
    most_frequent_rtt = df['Avg RTT DL (ms)'].value_counts().head(10)

    # RTT Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg RTT DL (ms)', y='MSISDN/Number', data=top_rtt, palette='Greens_d')
    plt.title('Top 10 RTT (Avg RTT DL)')
    plt.xlabel('Avg RTT DL (ms)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg RTT DL (ms)', y='MSISDN/Number', data=bottom_rtt, palette='Oranges_d')
    plt.title('Bottom 10 RTT (Avg RTT DL)')
    plt.xlabel('Avg RTT DL (ms)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    most_frequent_rtt.plot(kind='bar', color='cyan')
    plt.title('Most Frequent RTT (Avg RTT DL)')
    plt.xlabel('Avg RTT DL (ms)')
    plt.ylabel('Frequency')
    plt.show()

    # Top 10 Throughput (Avg Throughput DL)
    top_throughput = df.nlargest(10, 'Avg Bearer TP DL (kbps)')
    bottom_throughput = df.nsmallest(10, 'Avg Bearer TP DL (kbps)')
    most_frequent_throughput = df['Avg Bearer TP DL (kbps)'].value_counts().head(10)

    # Throughput Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=top_throughput, palette='Blues_d')
    plt.title('Top 10 Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=bottom_throughput, palette='Reds_d')
    plt.title('Bottom 10 Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    most_frequent_throughput.plot(kind='bar', color='purple')
    plt.title('Most Frequent Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('Frequency')
    plt.show()
    

def visualize_throughput(df):
    # Top 10 Throughput (Avg Bearer TP DL)
    top_throughput = df.nlargest(10, 'Avg Bearer TP DL (kbps)')
    bottom_throughput = df.nsmallest(10, 'Avg Bearer TP DL (kbps)')
    most_frequent_throughput = df['Avg Bearer TP DL (kbps)'].value_counts().head(10)

    # Throughput Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=top_throughput, palette='Purples_d')
    plt.title('Top 10 Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Avg Bearer TP DL (kbps)', y='MSISDN/Number', data=bottom_throughput, palette='Reds_d')
    plt.title('Bottom 10 Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('MSISDN/Number')
    plt.show()

    plt.figure(figsize=(10, 6))
    most_frequent_throughput.plot(kind='bar', color='green')
    plt.title('Most Frequent Throughput (Avg Bearer TP DL)')
    plt.xlabel('Avg Bearer TP DL (kbps)')
    plt.ylabel('Frequency')
    plt.show()
    

def visualize_throughput_and_tcp_per_handset(df):
    # Average Throughput per Handset Type
    throughput_per_handset = df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values()
    plt.figure(figsize=(12, 6))
    throughput_per_handset.plot(kind='bar', color='skyblue')
    plt.title('Average Throughput per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average Throughput (kbps)')
    plt.xticks(rotation=45)
    plt.show()

    # Average TCP Retransmission per Handset Type
    tcp_per_handset = df.groupby('Handset Type')['DL TP < 50 Kbps (%)'].mean().sort_values()
    plt.figure(figsize=(12, 6))
    tcp_per_handset.plot(kind='bar', color='lightcoral')
    plt.title('Average TCP Retransmission per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average TCP Retransmission (%)')
    plt.xticks(rotation=45)
    plt.show()
    

def process_and_cluster_customers(df, n_clusters=3):
    """
    Process customer data, normalize it, and perform K-means clustering.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing raw customer metrics.
    - n_clusters (int): Number of clusters for K-means.
    
    Returns:
    - pd.DataFrame: DataFrame with cluster assignments.
    """
    # Step 1: Aggregate customer metrics
    customer_aggregates = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'  # Assuming customers have one handset type
    }).reset_index()

    # Rename columns for clarity
    customer_aggregates.rename(columns={
        'Avg RTT DL (ms)': 'Avg RTT DL',
        'Avg RTT UL (ms)': 'Avg RTT UL',
        'Avg Bearer TP DL (kbps)': 'Avg Throughput DL',
        'Avg Bearer TP UL (kbps)': 'Avg Throughput UL',
    }, inplace=True)
    
    customer_aggregates['Avg RTT'] = (customer_aggregates['Avg RTT DL'] + customer_aggregates['Avg RTT UL']) / 2
    customer_aggregates['Avg Throughput'] = (customer_aggregates['Avg Throughput DL'] + customer_aggregates['Avg Throughput UL']) / 2

    # Step 2: Normalize data
    experience_metrics = customer_aggregates[['Avg RTT', 'Avg Throughput']]
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(experience_metrics)

    # Step 3: Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    customer_aggregates['Cluster'] = kmeans.fit_predict(scaled_metrics)

    # Describe each cluster
    for cluster in customer_aggregates['Cluster'].unique():
        print(f'Cluster {cluster}:')
        print(customer_aggregates[customer_aggregates['Cluster'] == cluster].describe())
    
    return customer_aggregates