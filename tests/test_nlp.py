import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN

# Import the functions to test
from src.pandas_survey_toolkit.analytics import fit_umap, fit_cluster_hdbscan
from src.pandas_survey_toolkit.nlp import fit_sentence_transformer

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


