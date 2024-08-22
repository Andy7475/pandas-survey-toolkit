import pandas as pd
import pandas_flavor as pf
from sentence_transformers import SentenceTransformer
import numpy as np
from pandas_survey_toolkit.utils import create_masked_df, combine_results

@pf.register_dataframe_method
def fit_sentence_transformer(df, input_column:str, model_name='all-MiniLM-L6-v2', output_column="sentence_embedding"):
    """Adds a list of vector embeddings for each string in the input column. These can then be used for downstream
    tasks like clustering"""
    # Initialize the sentence transformer model
    masked_df, mask = create_masked_df(df, [input_column])
    model = SentenceTransformer(model_name)
    
    # Create sentence embeddings
    embeddings = model.encode(masked_df[input_column].tolist())
    
    # Convert embeddings to a list of numpy arrays
    embeddings_list = [embedding for embedding in embeddings]
    
    # Add the embeddings as a new column in the dataframe
    masked_df[output_column] = embeddings_list
    df_to_return = combine_results(df, masked_df, mask, output_column)
    
    return df_to_return