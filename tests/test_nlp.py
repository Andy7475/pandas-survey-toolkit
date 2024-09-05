import numpy as np
import pandas as pd
import pytest
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from umap import UMAP

from .context import pandas_survey_toolkit

# Import the functions to test
from pandas_survey_toolkit.nlp import (encode_likert, extract_sentiment,
                                       fit_sentence_transformer, fit_spacy, cluster_questions)



@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        'Q1': np.random.choice(['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree'], 100),
        'Q2': np.random.choice(['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree'], 100),
        'Q3': np.random.choice(['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree'], 100),
        'Q4': np.random.choice(['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree'], 100),
        'OtherColumn': np.random.rand(100)
    })

def test_cluster_questions_columns(sample_df):
    result = sample_df.cluster_questions(columns=['Q1', 'Q2', 'Q3', 'Q4'])
    assert 'question_cluster_id' in result.columns
    assert 'question_cluster_probability' in result.columns
    assert 'umap_x' in result.columns
    assert 'umap_y' in result.columns

def test_cluster_questions_pattern(sample_df):
    result = sample_df.cluster_questions(pattern='^Q')
    assert 'question_cluster_id' in result.columns
    assert 'question_cluster_probability' in result.columns
    assert 'umap_x' in result.columns
    assert 'umap_y' in result.columns

def test_cluster_questions_custom_mapping(sample_df):
    custom_mapping = {
        'strongly agree': 2,
        'agree': 1,
        'neutral': 0,
        'disagree': -1,
        'strongly disagree': -2
    }
    result = sample_df.cluster_questions(columns=['Q1', 'Q2', 'Q3', 'Q4'], likert_mapping=custom_mapping)
    assert 'question_cluster_id' in result.columns
    assert 'question_cluster_probability' in result.columns

def test_cluster_questions_umap_parameters(sample_df):
    result = sample_df.cluster_questions(columns=['Q1', 'Q2', 'Q3', 'Q4'], 
                                         umap_n_neighbors=10, umap_min_dist=0.05)
    assert 'umap_x' in result.columns
    assert 'umap_y' in result.columns

def test_cluster_questions_hdbscan_parameters(sample_df):
    result = sample_df.cluster_questions(columns=['Q1', 'Q2', 'Q3', 'Q4'], 
                                         hdbscan_min_cluster_size=10, hdbscan_min_samples=5)
    assert 'question_cluster_id' in result.columns
    assert 'question_cluster_probability' in result.columns

def test_cluster_questions_error_no_columns_or_pattern(sample_df):
    with pytest.raises(ValueError):
        sample_df.cluster_questions()


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

def test_fit_spacy():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'comments': ['This is a test', 'Another comment', np.nan, 'SpaCy is cool']
    })
    
    # Apply the fit_spacy function
    result = fit_spacy(df, input_column='comments')
    
    # Check if the new column exists
    assert 'spacy_output' in result.columns
    
    # Check if the number of rows is preserved
    assert len(result) == len(df)
    
    # Check if the entries are spaCy Doc objects (where not NaN)
    for i, row in result.iterrows():
        if pd.notna(row['comments']):
            assert isinstance(row['spacy_output'], spacy.tokens.doc.Doc)
        else:
            assert pd.isna(row['spacy_output'])
    
    # Check if the content of the spaCy Doc objects matches the input
    nlp = spacy.load("en_core_web_md")
    for i, row in result.iterrows():
        if pd.notna(row['comments']):
            assert row['spacy_output'].text == row['comments']
            assert row['spacy_output'].text == nlp(row['comments']).text



@pytest.fixture
def sample_df2():
    return pd.DataFrame({
        'Q1': ['Strongly Agree', 'Disagree', 'Neither Agree nor Disagree', 'Agree', 'Strongly Disagree'],
        'Q2': ['Agree', 'Disagree', 'Neutral', 'Strongly Agree', 'Do not agree'],
        'Q3': ['Strongly Agree', np.nan, 'Neutral', 'Agree', 'Unconverted'],
        'Q4' : ['Very Satisfied', 'neither satisfied nor dissatisfied', 'dissatisfied', 'very dis-satisfied', 'satisfied']
    })

@pytest.fixture
def custom_mapping():
    return {
        'strongly agree': 2,
        'agree': 1,
        'neither agree nor disagree': 0,
        'neutral': 0,
        'disagree': -1,
        'strongly disagree': -2,
        'do not agree': -1
    }

def test_default_mapping(sample_df2):
    result = sample_df2.encode_likert(['Q1', 'Q2','Q4'])
    
    expected_Q1 = [1, -1, 0, 1, -1]
    expected_Q2 = [1, -1, 0, 1, -1]
    expected_Q4 = [1, 0, -1, -1, 1]
    
    assert list(result['likert_encoded_Q1']) == expected_Q1
    assert list(result['likert_encoded_Q2']) == expected_Q2
    assert list(result['likert_encoded_Q4']) == expected_Q4

def test_column_production(sample_df2):
    result = sample_df2.encode_likert(['Q1', 'Q2', 'Q3'])
    
    expected_columns = set(sample_df2.columns) | {'likert_encoded_Q1', 'likert_encoded_Q2', 'likert_encoded_Q3'}
    assert set(result.columns) == expected_columns

def test_nan_handling(sample_df2):
    result = sample_df2.encode_likert(['Q3'])
    
    assert pd.isna(result.loc[1, 'likert_encoded_Q3'])
    assert result.loc[0, 'likert_encoded_Q3'] == 1  # 'Strongly Agree'
    assert result.loc[2, 'likert_encoded_Q3'] == 0  # 'Neutral'

def test_custom_mapping(sample_df2, custom_mapping):
    result = sample_df2.encode_likert(['Q1', 'Q2'], custom_mapping=custom_mapping)
    
    expected_Q1 = [2, -1, 0, 1, -2]
    expected_Q2 = [1, -1, 0, 2, -1]
    
    assert list(result['likert_encoded_Q1']) == expected_Q1
    assert list(result['likert_encoded_Q2']) == expected_Q2

def test_custom_mapping_nan_handling(sample_df2, custom_mapping):
    result = sample_df2.encode_likert(['Q3'], custom_mapping=custom_mapping)
    
    assert pd.isna(result.loc[1, 'likert_encoded_Q3'])
    assert result.loc[0, 'likert_encoded_Q3'] == 2  # 'Strongly Agree'
    assert result.loc[2, 'likert_encoded_Q3'] == 0  # 'Neutral'

def test_output_prefix(sample_df2):
    result = sample_df2.encode_likert(['Q1'], output_prefix='custom_')
    
    assert 'custom_Q1' in result.columns
    assert 'likert_encoded_Q1' not in result.columns

def test_unconverted_warning(sample_df2, custom_mapping):
    with pytest.warns(UserWarning, match="The following phrases were not converted"):
        sample_df2.encode_likert(['Q3'], custom_mapping=custom_mapping)

def test_default_mapping_warning(sample_df2):
    with pytest.warns(UserWarning, match="The default mapping didn't convert the following responses"):
        sample_df2.encode_likert(['Q3'])