#!/usr/bin/env python
# coding: utf-8

# In[13]:


from pathlib import Path
import pandas as pd
import numpy as np

def count_csv_rows(file_path, has_header=True):
    """
    Counts the number of rows in a CSV file.

    Parameters:
    - file_path (Path): The path to the CSV file.
    - has_header (bool): Whether the CSV file has a header row.

    Returns:
    - int: The number of data rows in the CSV file.
    """
    with file_path.open('r', encoding='utf-8') as file:
        row_count = sum(1 for _ in file)
    return row_count - 1 if has_header else row_count

def enumerate_and_sort_csv(datadir, has_header=True):
    """
    Enumerates CSV files in the given directory, counts their rows, and sorts them.

    Parameters:
    - datadir (str or Path): The directory containing CSV files.
    - has_header (bool): Whether the CSV files have header rows.

    Returns:
    - List of tuples: Each tuple contains (file_name, row_count), sorted by row_count ascending.
    """
    datadir = Path(datadir)
    if not datadir.is_dir():
        raise ValueError(f"The path {datadir} is not a valid directory.")

    # Find all CSV files in the directory
    csv_files = list(datadir.glob('*.csv'))

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return []

    file_row_counts = []

    for file in csv_files:
        try:
            rows = count_csv_rows(file, has_header=has_header)
            file_row_counts.append((file.name, rows))
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Sort the list of tuples based on row_count (ascending order)
    sorted_files = sorted(file_row_counts, key=lambda x: x[1])

    return sorted_files




# In[14]:


# Specify the directory containing CSV files
data_directory = 'data/prices'  # Replace with your directory path

# Enumerate and sort CSV files
sorted_csv_files = enumerate_and_sort_csv(data_directory, has_header=True)

if sorted_csv_files:
    print("\nCSV files sorted by number of rows (ascending):")
    for filename, count in sorted_csv_files:
        print(f"{filename}: {count} rows")


# In[8]:


from collections import defaultdict
import numpy as np

def cluster_csv_pairs_hdbscan_sorted(sorted_csv_files,
                                     min_cluster_size=5,
                                     min_samples=None,
                                     alpha=1.0):
    """
    HDBSCAN clustering with cluster IDs reordered by descending row counts.
    Cluster 0 has the largest rows.
    """

    import hdbscan
    from sklearn.preprocessing import RobustScaler

    if not sorted_csv_files:
        return {}

    # Feature: row count only
    row_counts = np.array([rc for _, rc in sorted_csv_files], dtype=np.float64).reshape(-1, 1)

    # Robust scaling
    X = RobustScaler().fit_transform(row_counts)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        alpha=alpha,
        metric="euclidean"
    )

    raw_labels = clusterer.fit_predict(X)

    # Collect raw clusters
    raw_clusters = defaultdict(list)
    for pair, lab in zip(sorted_csv_files, raw_labels):
        raw_clusters[int(lab)].append(pair)

    # Separate noise
    noise = raw_clusters.pop(-1, [])

    # Sort clusters by mean row count (descending)
    sorted_clusters = sorted(
        raw_clusters.values(),
        key=lambda items: np.mean([rc for _, rc in items]),
        reverse=True
    )

    # Reassign cluster IDs: 0 = largest rows
    final_clusters = {}
    for new_id, items in enumerate(sorted_clusters):
        final_clusters[new_id] = sorted(items, key=lambda x: x[1], reverse=True)

    if noise:
        final_clusters[-1] = sorted(noise, key=lambda x: x[1], reverse=True)

    # Pretty print
    print(f"Clusters found (excluding noise): {len(sorted_clusters)}")
    print(f"Noise points (-1): {len(noise)}\n")

    for cid in sorted(final_clusters.keys(), key=lambda x: (x == -1, x)):
        title = "Noise (-1)" if cid == -1 else f"Cluster {cid}"
        print(f"{title} | size={len(final_clusters[cid])}")
        for fname, rc in final_clusters[cid]:
            print(f"  ({fname!r}, {rc})")
        print()

    return final_clusters


# ---- Example usage ----
# clusters = cluster_csv_pairs_hdbscan_sorted(
#     sorted_csv_files,
#     min_cluster_size=5
# )


# In[15]:


from collections import defaultdict
import numpy as np

def cluster_csv_pairs_k_mean_sorted(sorted_csv_files,
                                    n_clusters,
                                    random_state=None):
    """
    K-Means clustering with cluster IDs reordered by descending row counts.
    Cluster 0 has the largest rows, then 1, etc.

    Parameters
    ----------
    sorted_csv_files : List[Tuple[str, int]]
        A list of (filename, row_count) pairs.
    n_clusters : int
        The number of clusters to form.
    random_state : int or None
        Random seed for KMeans initialization.

    Returns
    -------
    dict[int, List[Tuple[str, int]]]
        A dict mapping new_cluster_id -> list of (filename, row_count),
        with filenames sorted by row count descending.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler

    if not sorted_csv_files:
        return {}

    # Extract row counts and reshape for sklearn
    row_counts = np.array([rc for _, rc in sorted_csv_files], dtype=np.float64).reshape(-1, 1)

    # Robust scaling to reduce influence of outliers
    X = RobustScaler().fit_transform(row_counts)

    # Run KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    labels = kmeans.fit_predict(X)

    # Collect raw clusters
    raw_clusters = defaultdict(list)
    for (fname, rc), lab in zip(sorted_csv_files, labels):
        raw_clusters[int(lab)].append((fname, rc))

    # Re‐order clusters by descending mean row count
    sorted_cluster_items = sorted(
        raw_clusters.values(),
        key=lambda items: np.mean([rc for _, rc in items]),
        reverse=True
    )

    # Assign new IDs: 0 = largest mean row count, 1 = next largest, ...
    final_clusters = {}
    for new_id, items in enumerate(sorted_cluster_items):
        # Within each cluster sort by row count descending
        final_clusters[new_id] = sorted(items, key=lambda x: x[1], reverse=True)

    # Pretty print
    print(f"KMeans Clusters found: {len(final_clusters)}\n")
    for cid in sorted(final_clusters.keys()):
        print(f"Cluster {cid} | size={len(final_clusters[cid])}")
        for fname, rc in final_clusters[cid]:
            print(f"  ({fname!r}, {rc})")
        print()

    return final_clusters


# ---- Example usage ----
# Suppose you have:
# sorted_csv_files = [
#     ("a.csv", 150),
#     ("b.csv", 2000),
#     ("c.csv", 75),
#     # ...
# ]
# clusters = cluster_csv_pairs_k_mean_sorted(
#     sorted_csv_files,
#     n_clusters=4,
#     random_state=42
# )


# In[16]:


clusters = cluster_csv_pairs_k_mean_sorted(sorted_csv_files,
                                              n_clusters=10,
                                              random_state=42)


# In[ ]:


output_dir = "input_files"
# create if it does not exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

for cluster_id in range(len(clusters)-1):
    print(f"Processing cluster {cluster_id}")
    dataframes_vol = []
    for filename, rows in clusters[cluster_id]:
        print(f"File: {filename}, Rows: {rows}")
        # make one single pandas dataframe contains the data from above stock IDs, the index is the data, the columns are the stock IDs
        stock_id = filename[:-4]  

        file_path = Path(data_directory) / filename
        df_vol = pd.read_csv(file_path, usecols=['Date', 'Adj Close', 'Volume'], parse_dates=['Date'])
        df_vol.set_index('Date', inplace=True)
        df_vol.rename(columns={'Adj Close': f"{stock_id}_price", 'Volume': f"{stock_id}_vol"}, inplace=True)
        dataframes_vol.append(df_vol)

    combined_df_vol = pd.concat(dataframes_vol, axis=1)
    combined_df_vol.fillna(method='ffill', inplace=True)
    combined_df_vol.fillna(method='bfill', inplace=True)
    log_returns_df_vol = np.log(combined_df_vol / (combined_df_vol.shift(1) + 1e-9 ) + 1e-9)
    # drop the first row
    #log_returns_df_vol = log_returns_df_vol.iloc[1:]
    log_returns_df_vol = log_returns_df_vol.dropna()  # Drop the first row with

    price_columns = [col for col in log_returns_df_vol.columns if col.endswith('_price')]
    volume_columns = [col for col in log_returns_df_vol.columns if col.endswith('_vol')]
    log_returns_df_vol = log_returns_df_vol[price_columns + volume_columns]
    # find out the earliest date in the dataframe
    start_date = log_returns_df_vol.index.min().strftime('%Y-%m-%d')
    log_returns_df_vol.to_csv(f"{output_dir}/stock_data_vol-cluster-{cluster_id}-startdate-{start_date}.csv")

