from typing import List, Union
import pandas as pd
import numpy as np

def create_masked_df(df, input_columns):
    """
    Create a masked DataFrame excluding rows with NaN values in specified columns.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_columns (list): List of column names to check for NaN values.
    
    Returns:
    tuple: (masked_df, mask)
        masked_df (pandas.DataFrame): DataFrame with NaN rows removed.
        mask (pandas.Series): Boolean mask indicating non-NaN rows.
    """
    mask = df[input_columns].notna().all(axis=1)
    masked_df = df[mask].copy()
    return masked_df, mask

def combine_results(original_df, result_df, mask, output_columns:Union[List[str], str]):
    """
    Combine the results from a function applied to a masked DataFrame 
    back into the original DataFrame.
    
    Parameters:
    original_df (pandas.DataFrame): The original input DataFrame.
    result_df (pandas.DataFrame): The DataFrame with results to be combined.
    mask (pandas.Series): Boolean mask indicating which rows to update.
    output_columns (list or str): List of column names (or name of single column) for the output.
    
    Returns:
    pandas.DataFrame: The original DataFrame updated with new results.
    """

    df = original_df.copy()
    
    if isinstance(output_columns, str):
        output_columns = [output_columns]
    
    df.loc[mask, output_columns] = result_df.loc[mask, output_columns]
    
    return df

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from typing import Tuple

def apply_vectorizer(df: pd.DataFrame, input_column: str, vectorizer_name: str = 'TfidfVectorizer', feature_prefix: str = 'vect_features_', **vectorizer_kwargs) -> Tuple[pd.DataFrame, object, np.ndarray]:
    """
    Apply a vectorizer to a text column in a DataFrame and return a new DataFrame with the vectorized features,
    along with the fitted vectorizer and feature names.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to vectorize.
    vectorizer_name (str): Name of the vectorizer to use ('CountVectorizer' or 'TfidfVectorizer'). Default is 'TfidfVectorizer'.
    feature_prefix (str): Prefix for the feature column names. Default is 'vect_features_'.
    **vectorizer_kwargs: Additional keyword arguments to pass to the vectorizer.

    Returns:
    Tuple[pandas.DataFrame, object, np.ndarray]: 
        - A new DataFrame containing the vectorized features
        - The fitted vectorizer object
        - An array of feature names
    """
    # Select the appropriate vectorizer
    if vectorizer_name == 'CountVectorizer':
        vectorizer = CountVectorizer(**vectorizer_kwargs)
    elif vectorizer_name == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    else:
        raise ValueError("Unsupported vectorizer. Use 'CountVectorizer' or 'TfidfVectorizer'.")

    # Fit and transform the input text
    feature_matrix = vectorizer.fit_transform(df[input_column].fillna(''))

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame with the vectorized features
    feature_df = pd.DataFrame(
        feature_matrix.toarray(),
        columns=[f"{feature_prefix}{name}" for name in feature_names],
        index=df.index
    )

    return feature_df, vectorizer, feature_names