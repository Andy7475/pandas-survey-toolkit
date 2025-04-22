import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from pandas_survey_toolkit.utils import combine_results, create_masked_df


@pf.register_dataframe_method
def fit_umap(
    df, input_columns: Union[List[str], str], output_columns=["umap_x", "umap_y"], target_y:str=None, embeddings_in_list=False, **kwargs
):
    """Apply UMAP to the columns in the dataframe.
    
    This function applies UMAP dimensionality reduction to the specified columns
    and appends the x and y coordinates to the dataframe as new columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to transform.
    input_columns : Union[List[str], str]
        Column name(s) containing the data to reduce.
    output_columns : list, optional
        Names for the output coordinate columns, by default ["umap_x", "umap_y"]
    target_y : str, optional
        Name of a column to use as the target variable for supervised UMAP, by default None
    embeddings_in_list : bool, optional
        Set to True if embeddings are a list of values in a single column,
        False if each column is a separate dimension, by default False
    **kwargs
        Additional arguments to pass to UMAP. Most important is n_neighbors (default is 15).
    
    Returns
    -------
    pandas.DataFrame
        The input dataframe with added UMAP coordinate columns.
    
    Raises
    ------
    KeyError
        If the specified target_y is not a column in the dataframe.
    ValueError
        If embeddings_in_list is True but multiple input columns are provided.
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
    """Apply HDBSCAN clustering to the specified columns of the DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_columns : list, optional
        List of column names to use for clustering, by default ['umap_x', 'umap_y']
    output_columns : list, optional
        Names for the output columns, by default ["cluster", "cluster_probability"]
    min_cluster_size : int, optional
        The minimum size of clusters, by default 5
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered a core point, by default None
    cluster_selection_epsilon : float, optional
        A distance threshold. Clusters below this value will be merged.
        Higher epsilon means fewer, larger clusters, by default 0.0
    metric : str, optional
        The metric to use for distance computation, by default 'euclidean'
    cluster_selection_method : str, optional
        The method to select clusters. Either 'eom' or 'leaf', by default 'eom'
    allow_single_cluster : bool, optional
        Whether to allow a single cluster, by default False
    
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns containing cluster labels and probabilities.
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


