import pandas as pd
import pandas_flavor as pf
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import pandas as pd
import pandas_flavor as pf
from sklearn.cluster import HDBSCAN

@pf.register_dataframe_method
def fit_umap(df, embedding_column='sentence_embedding', 
             n_neighbors=15, 
             n_components=2, metric='cosine', random_state=42):
    """
    Apply UMAP to the embeddings in the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    embedding_column (str): Name of the column to store embeddings. Default is 'sentence_embedding'.
    n_neighbors (int): Number of neighbors to consider in UMAP. Default is 15.
    n_components (int): Number of dimensions to reduce to. Default is 2.
    metric (str): The metric to use for UMAP. Default is 'cosine'.
    random_state (int): Random state for reproducibility. Default is 42.
    
    Returns:
    pandas.DataFrame: The input DataFrame with additional UMAP coordinate columns.
    """
   
    # Get embeddings as a numpy array
    embeddings = np.array(df[embedding_column].tolist())
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, 
                             metric=metric, random_state=random_state)
    
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    
    # Add UMAP coordinates to the DataFrame
    for i in range(n_components):
        df[f'umap_{i+1}'] = umap_embeddings[:, i]
    
    return df

@pf.register_dataframe_method
def fit_cluster_hdbscan(df, input_columns=['umap_1', 'umap_2'], output_columns=["cluster", "cluster_probability"], min_cluster_size=5, min_samples=None, 
                        cluster_selection_epsilon=0.0, metric='euclidean', cluster_selection_method='eom',
                        allow_single_cluster=False):
    """
    Apply HDBSCAN clustering to the specified columns of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): List of column names to use for clustering. Default is ['umap_1', 'umap_2'].
    min_cluster_size (int): The minimum size of clusters. Default is 5.
    min_samples (int): The number of samples in a neighborhood for a point to be considered a core point. Default is None.
    cluster_selection_epsilon (float): A distance threshold. Clusters below this value will be merged. Default is 0.0.
    metric (str): The metric to use for distance computation. Default is 'euclidean'.
    cluster_selection_method (str): The method to select clusters. Either 'eom' or 'leaf'. Default is 'eom'.
    allow_single_cluster (bool): Whether to allow a single cluster. Default is False.
    
    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'cluster' column containing cluster labels.
    """
    # Extract the specified columns for clustering

    X = df[input_columns].values
    
    # Initialize and fit HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        metric=metric,
                        cluster_selection_method=cluster_selection_method,
                        allow_single_cluster=allow_single_cluster)
    
    cluster_labels = clusterer.fit_predict(X)
    
    # Add cluster labels to the DataFrame
    df[output_columns[0]] = cluster_labels
    
    # Add cluster probabilities to the DataFrame
    df[output_columns[1]] = clusterer.probabilities_
    
    
    return df