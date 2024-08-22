import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.datasets import make_blobs
from umap import UMAP
from sklearn.cluster import HDBSCAN

from .context import pandas_survey_toolkit

# Import the functions to test
from pandas_survey_toolkit.analytics import fit_umap, fit_cluster_hdbscan
from pandas_survey_toolkit.nlp import fit_sentence_transformer

# Test for fit_umap
def test_fit_umap():
    # Create a sample DataFrame with embeddings
    n_samples = 300
    n_features = 100
    n_clusters = 3
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    
    # Convert to list of arrays and add a NaN row
    embeddings = [arr for arr in X]
    embeddings.insert(1, np.nan)  # Insert NaN as the second element
    
    df = pd.DataFrame({
        'sentence_embedding': embeddings
    })
    
    result = df.fit_umap(input_columns="sentence_embedding", embeddings_in_list=True)
    
    assert 'umap_x' in result.columns
    assert 'umap_y' in result.columns
    assert len(result) == 301
    assert np.isnan(result['umap_x'][1]) and np.isnan(result['umap_y'][1])  # Check if NaN is preserved

# Test for fit_cluster_hdbscan
def test_fit_cluster_hdbscan():
    df = pd.DataFrame({
        'umap_x': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        'umap_y': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
    }, dtype=np.float64)
    
    result = fit_cluster_hdbscan(df)
    
    assert 'cluster' in result.columns
    assert 'cluster_probability' in result.columns
    assert result.shape == (6, 4) #2 extra columns added
    assert np.isnan(result['cluster'][2])  # Check if NaN is preserved
    assert np.isnan(result['cluster_probability'][2])  # Check if NaN is preserved