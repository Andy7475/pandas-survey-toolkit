import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pandas_survey_toolkit.utils import combine_results, create_masked_df


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

@pf.register_dataframe_method
def extract_sentiment(df, input_column: str, output_columns=["positive", "neutral", "negative", "sentiment"]):
    """
    Extract sentiment from text using the cardiffnlp/twitter-roberta-base-sentiment model.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to analyze.
    output_columns (list): List of column names for the output. Default is ["positive", "neutral", "negative", "sentiment"].
    
    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for sentiment scores and labels.
    """
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    masked_df, mask = create_masked_df(df, [input_column])
    
    def analyze_sentiment(text):
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        return scores
    
    sentiment_scores = masked_df[input_column].apply(analyze_sentiment)
    
    masked_df[output_columns[0]] = sentiment_scores.apply(lambda x: x[2])  # Positive
    masked_df[output_columns[1]] = sentiment_scores.apply(lambda x: x[1])  # Neutral
    masked_df[output_columns[2]] = sentiment_scores.apply(lambda x: x[0])  # Negative
    
    masked_df[output_columns[3]] = masked_df[[output_columns[0], output_columns[1], output_columns[2]]].idxmax(axis=1)
    masked_df[output_columns[3]] = masked_df[output_columns[3]].map({output_columns[0]: 'positive', 
                                                                     output_columns[1]: 'neutral', 
                                                                     output_columns[2]: 'negative'})
    
    df_to_return = combine_results(df, masked_df, mask, output_columns)
    return df_to_return