import pandas as pd
import pandas_flavor as pf
from sentence_transformers import SentenceTransformer
import numpy as np

@pf.register_dataframe_method
def fit_sentence_transformer(df, input_column:str, model_name='all-MiniLM-L6-v2', output_column="sentence_embedding"):
    """Adds a list of vector embeddings for each string in the input column. These can then be used for downstream
    tasks like clustering"""
    # Initialize the sentence transformer model
    df_temp = df.copy()
    model = SentenceTransformer(model_name)
    
    # Create sentence embeddings
    embeddings = model.encode(df_temp[input_column].tolist())
    
    # Convert embeddings to a list of numpy arrays
    embeddings_list = [np.array(embedding) for embedding in embeddings]
    
    # Add the embeddings as a new column in the dataframe
    df_temp[output_column] = embeddings_list
    
    return df_temp