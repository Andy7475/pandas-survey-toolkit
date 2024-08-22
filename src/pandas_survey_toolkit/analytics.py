from typing import Union, List
import warnings
import pandas as pd
import pandas_flavor as pf
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import pandas as pd
import pandas_flavor as pf
from sklearn.cluster import HDBSCAN
from pandas_survey_toolkit.utils import create_masked_df, combine_results

@pf.register_dataframe_method
def fit_umap(
    df, input_columns: Union[List[str], str], output_columns=["umap_x", "umap_y"], target_y:str=None, embeddings_in_list=False, **kwargs
):
    """applies UMAP to the columns in the dataframe and appends the x and y co-ordinates
    to the dataframe as 2 new columns
    most import kwargs to use would be n_neighbors (default is 15) - note american spelling.
    If your embeddings are a list of values in a single column, set embeddings_in_list to True,
    otherwise it assumes each column is a separate set of values / dimension to be reduced.

    Returns: modified dataframe
    """

    if isinstance(input_columns, str):
        input_columns = [input_columns] #ensure consistent handling in code

    columns_to_mask = input_columns
    if target_y:
        if target_y not in df.columns:
            raise KeyError(f"Your target_y value {target_y} should be the name of a column in the dataframe.")
        columns_to_mask = input_columns + [target_y]

    masked_df, mask = create_masked_df(df, columns_to_mask) #propogate NaN

    if embeddings_in_list:
        if len(input_columns) > 1:
            raise ValueError("If your embeddings are in a list, they should be in a single column.")
        embedding_data = np.array(masked_df[input_columns[0]].tolist())
    else:
        embedding_data = masked_df[input_columns].values

     # Adjust n_neighbors if the dataset is too small
    original_n_neighbors = kwargs.get('n_neighbors', 15)
    adjusted_n_neighbors = min(original_n_neighbors, max(2, embedding_data.shape[0] - 1))
    
    if adjusted_n_neighbors != original_n_neighbors:
        warnings.warn(f"n_neighbors adjusted from {original_n_neighbors} to {adjusted_n_neighbors} due to small dataset size.")
    
    kwargs['n_neighbors'] = adjusted_n_neighbors
    
    reducer = umap.UMAP(**kwargs)
    if target_y is not None:
        target_y = masked_df[target_y].values

    umap_coordinates = reducer.fit_transform(embedding_data, target_y)

    # Append UMAP coordinates to DataFrame
    masked_df[output_columns[0]] = umap_coordinates[:, 0]
    masked_df[output_columns[1]] = umap_coordinates[:, 1]

    df_to_return = combine_results(df, masked_df, mask, output_columns)
    return df_to_return

@pf.register_dataframe_method
def fit_cluster_hdbscan(df, input_columns=['umap_x', 'umap_y'], output_columns=["cluster", "cluster_probability"], min_cluster_size=5, min_samples=None, 
                        cluster_selection_epsilon=0.0, metric='euclidean', cluster_selection_method='eom',
                        allow_single_cluster=False):
    """
    Apply HDBSCAN clustering to the specified columns of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): List of column names to use for clustering. Default is ['umap_1', 'umap_2'].
    min_cluster_size (int): The minimum size of clusters. Default is 5.
    min_samples (int): The number of samples in a neighborhood for a point to be considered a core point. Default is None.
    cluster_selection_epsilon (float): A distance threshold. Clusters below this value will be merged. Default is 0.0. higher epslion = fewer, larger clusters
    metric (str): The metric to use for distance computation. Default is 'euclidean'.
    cluster_selection_method (str): The method to select clusters. Either 'eom' or 'leaf'. Default is 'eom'.
    allow_single_cluster (bool): Whether to allow a single cluster. Default is False.
    
    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'cluster' column containing cluster labels.
    """
    # Extract the specified columns for clustering

    masked_df, mask = create_masked_df(df, input_columns)

    X = masked_df[input_columns].values
    
    # Initialize and fit HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        metric=metric,
                        cluster_selection_method=cluster_selection_method,
                        allow_single_cluster=allow_single_cluster)
    
    cluster_labels = clusterer.fit_predict(X)
    
    # Add cluster labels to the DataFrame
    masked_df[output_columns[0]] = cluster_labels
    
    # Add cluster probabilities to the DataFrame
    masked_df[output_columns[1]] = clusterer.probabilities_
    
    df_to_return = combine_results(df, masked_df, mask, output_columns)
    return df_to_return