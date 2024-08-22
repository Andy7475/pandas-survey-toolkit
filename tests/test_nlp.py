import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN

from .context import pandas_survey_toolkit

# Import the functions to test
from pandas_survey_toolkit.analytics import fit_umap, fit_cluster_hdbscan
from pandas_survey_toolkit.nlp import fit_sentence_transformer, extract_sentiment

# Test for fit_sentence_transformer
def test_fit_sentence_transformer():
    df = pd.DataFrame({
        'text': ['Hello world', 'Test sentence', np.nan, 'Another test'],
    })
    
    result = fit_sentence_transformer(df, input_column='text')
    
    assert 'sentence_embedding' in result.columns
    assert len(result) == 4
    assert isinstance(result['sentence_embedding'][0], np.ndarray)
    assert np.isnan(result['sentence_embedding'][2]).all()  # Check if NaN is preserved


def test_extract_sentiment():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'text': ['This is amazing!', 'I hate this, it\'s awful', 'Neutral statement', np.nan, 'Another positive example']
    })
    
    # Apply the sentiment analysis
    result = df.extract_sentiment(input_column='text')
    
    # Check if new columns are added
    assert 'positive' in result.columns
    assert 'neutral' in result.columns
    assert 'negative' in result.columns
    assert 'sentiment' in result.columns
    
    # Check if the DataFrame has the correct shape
    assert result.shape == (5, 5)  # Original column + 4 new columns
    
    # Check positive sentiment
    assert result.loc[0, 'sentiment'] == 'positive'
    assert result.loc[0, 'positive'] > result.loc[0, 'negative']
    
    # Check negative sentiment
    assert result.loc[1, 'sentiment'] == 'negative'
    assert result.loc[1, 'negative'] > result.loc[1, 'positive']
    
    # Check if NaN is preserved
    assert np.isnan(result.loc[3, 'positive'])
    assert np.isnan(result.loc[3, 'neutral'])
    assert np.isnan(result.loc[3, 'negative'])
    assert pd.isna(result.loc[3, 'sentiment'])
    
    # Check if all sentiment scores are between 0 and 1
    assert ((result['positive'] >= 0) & (result['positive'] <= 1) | result['positive'].isna()).all()
    assert ((result['neutral'] >= 0) & (result['neutral'] <= 1) | result['neutral'].isna()).all()
    assert ((result['negative'] >= 0) & (result['negative'] <= 1) | result['negative'].isna()).all()
    
    # Check if sentiment labels are correct
    assert set(result['sentiment'].dropna().unique()) <= {'positive', 'neutral', 'negative'}
