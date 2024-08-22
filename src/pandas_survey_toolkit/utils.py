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