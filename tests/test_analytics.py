import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from pandas_survey_toolkit.analytics import fit_cluster_hdbscan


# Test for fit_umap
def test_fit_umap():
    # Create a sample DataFrame with embeddings
    n_samples = 300
    n_features = 100
    n_clusters = 3
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42
    )

    # Convert to list of arrays and add a NaN row
    embeddings = [arr for arr in X]
    embeddings.insert(1, np.nan)  # Insert NaN as the second element

    df = pd.DataFrame({"sentence_embedding": embeddings})

    result = df.fit_umap(input_columns="sentence_embedding", embeddings_in_list=True)

    assert "umap_x" in result.columns
    assert "umap_y" in result.columns
    assert len(result) == 301
    assert np.isnan(result["umap_x"][1]) and np.isnan(
        result["umap_y"][1]
    )  # Check if NaN is preserved


# Test for fit_cluster_hdbscan
def test_fit_cluster_hdbscan():
    df = pd.DataFrame(
        {
            "umap_x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "umap_y": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        },
        dtype=np.float64,
    )

    result = fit_cluster_hdbscan(df)

    assert "cluster" in result.columns
    assert "cluster_probability" in result.columns
    assert result.shape == (6, 4)  # 2 extra columns added
    assert np.isnan(result["cluster"][2])  # Check if NaN is preserved
    assert np.isnan(result["cluster_probability"][2])  # Check if NaN is preserved


# ---------------------------------------------------------------------------
# Additional fit_umap tests — covers previously uncovered branches
# ---------------------------------------------------------------------------


def test_fit_umap_multiple_input_columns():
    """Test with multiple flat columns (non-embedding-list mode)."""
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 51)] * 3,
            "y": [float(i) for i in range(51, 101)] * 3,
        }
    )
    result = df.fit_umap(input_columns=["x", "y"])
    assert "umap_x" in result.columns
    assert "umap_y" in result.columns
    assert len(result) == len(df)


def test_fit_umap_supervised_target_y():
    """Supervised UMAP — covers the target_y assignment branch (line 82)."""
    n = 120
    X, y = make_blobs(n_samples=n, n_features=10, centers=3, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["label"] = y
    result = df.fit_umap(input_columns=[f"f{i}" for i in range(10)], target_y="label")
    assert "umap_x" in result.columns
    assert "umap_y" in result.columns
    assert len(result) == n


def test_fit_umap_target_y_not_in_df():
    """Raises KeyError when target_y column does not exist."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    with pytest.raises(KeyError, match="nonexistent"):
        df.fit_umap(input_columns=["x"], target_y="nonexistent")


def test_fit_umap_embeddings_in_list_multiple_columns_raises():
    """Raises ValueError when embeddings_in_list=True with multiple columns."""
    df = pd.DataFrame(
        {
            "a": [[1, 2, 3]] * 10,
            "b": [[4, 5, 6]] * 10,
        }
    )
    with pytest.raises(ValueError, match="single column"):
        df.fit_umap(input_columns=["a", "b"], embeddings_in_list=True)


def test_fit_umap_small_dataset_warns_and_adjusts_n_neighbors():
    """n_neighbors is adjusted down (with a warning) when dataset is too small.
    8 rows gives n_neighbors=7 (< 15), which is large enough for UMAP to run."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    with pytest.warns(UserWarning, match="n_neighbors adjusted"):
        result = df.fit_umap(input_columns=["x", "y"], n_neighbors=15)
    assert "umap_x" in result.columns


def test_fit_umap_custom_output_columns():
    df = pd.DataFrame(
        {
            "x": list(range(50)),
            "y": list(range(50, 100)),
        }
    )
    result = df.fit_umap(input_columns=["x", "y"], output_columns=["dim1", "dim2"])
    assert "dim1" in result.columns
    assert "dim2" in result.columns


# ---------------------------------------------------------------------------
# Additional fit_cluster_hdbscan tests — covers custom params
# ---------------------------------------------------------------------------


def test_fit_cluster_hdbscan_custom_output_columns():
    df = pd.DataFrame(
        {
            "umap_x": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2, 9.3],
            "umap_y": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2, 9.3],
        }
    )
    result = df.fit_cluster_hdbscan(
        input_columns=["umap_x", "umap_y"],
        output_columns=["my_cluster", "my_prob"],
        min_cluster_size=2,
    )
    assert "my_cluster" in result.columns
    assert "my_prob" in result.columns


def test_fit_cluster_hdbscan_leaf_method_and_single_cluster():
    df = pd.DataFrame(
        {
            "umap_x": [1.0, 1.1, 1.2, 1.3, 1.4],
            "umap_y": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    result = df.fit_cluster_hdbscan(
        min_cluster_size=2,
        cluster_selection_method="leaf",
        allow_single_cluster=True,
    )
    assert "cluster" in result.columns
    assert "cluster_probability" in result.columns
