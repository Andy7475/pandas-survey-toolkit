"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from pandas_survey_toolkit.utils import (
    apply_vectorizer,
    combine_results,
    create_masked_df,
)

# ---------------------------------------------------------------------------
# create_masked_df
# ---------------------------------------------------------------------------


def test_create_masked_df_removes_nan_rows():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    masked_df, mask = create_masked_df(df, ["a", "b"])
    assert len(masked_df) == 1
    assert masked_df.index.tolist() == [0]
    assert mask.sum() == 1


def test_create_masked_df_no_nans():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    masked_df, mask = create_masked_df(df, ["a", "b"])
    assert len(masked_df) == 3
    assert mask.all()


def test_create_masked_df_single_column():
    df = pd.DataFrame({"text": ["hello", np.nan, "world"]})
    masked_df, mask = create_masked_df(df, ["text"])
    assert len(masked_df) == 2
    assert mask.sum() == 2


def test_create_masked_df_all_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    masked_df, mask = create_masked_df(df, ["a"])
    assert len(masked_df) == 0
    assert mask.sum() == 0


# ---------------------------------------------------------------------------
# combine_results
# ---------------------------------------------------------------------------


def test_combine_results_updates_masked_rows():
    original = pd.DataFrame({"a": [1, 2, 3]})
    result = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    mask = pd.Series([True, False, True])
    combined = combine_results(original, result, mask, ["b"])
    assert combined["b"][0] == 10
    assert pd.isna(combined["b"][1])  # not in mask — stays NaN
    assert combined["b"][2] == 30


def test_combine_results_single_string_column():
    original = pd.DataFrame({"a": [1, 2]})
    result = pd.DataFrame({"a": [1, 2], "label": ["x", "y"]})
    mask = pd.Series([True, True])
    combined = combine_results(original, result, mask, "label")
    assert list(combined["label"]) == ["x", "y"]


def test_combine_results_does_not_modify_original():
    original = pd.DataFrame({"a": [1, 2, 3]})
    original_copy = original.copy()
    result = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    mask = pd.Series([True, True, True])
    combine_results(original, result, mask, ["b"])
    pd.testing.assert_frame_equal(original, original_copy)


# ---------------------------------------------------------------------------
# apply_vectorizer
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = ["hello world", "world peace", "hello peace", "hello world peace"]


def test_apply_vectorizer_tfidf():
    df = pd.DataFrame({"text": SAMPLE_TEXTS})
    feature_df, vectorizer, feature_names = apply_vectorizer(
        df, "text", vectorizer_name="TfidfVectorizer"
    )
    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_names) > 0
    assert feature_df.shape[0] == len(SAMPLE_TEXTS)
    assert feature_df.shape[1] == len(feature_names)


def test_apply_vectorizer_count():
    df = pd.DataFrame({"text": SAMPLE_TEXTS})
    feature_df, vectorizer, feature_names = apply_vectorizer(
        df, "text", vectorizer_name="CountVectorizer"
    )
    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_names) > 0
    # CountVectorizer should give integer-valued (or float) counts >= 0
    assert (feature_df.values >= 0).all()


def test_apply_vectorizer_invalid_raises():
    df = pd.DataFrame({"text": SAMPLE_TEXTS})
    with pytest.raises(ValueError, match="Unsupported vectorizer"):
        apply_vectorizer(df, "text", vectorizer_name="BogusVectorizer")


def test_apply_vectorizer_feature_prefix():
    df = pd.DataFrame({"text": SAMPLE_TEXTS})
    feature_df, _, _ = apply_vectorizer(
        df, "text", vectorizer_name="TfidfVectorizer", feature_prefix="feat_"
    )
    assert all(col.startswith("feat_") for col in feature_df.columns)


def test_apply_vectorizer_preserves_index():
    df = pd.DataFrame({"text": SAMPLE_TEXTS}, index=[10, 20, 30, 40])
    feature_df, _, _ = apply_vectorizer(df, "text")
    assert list(feature_df.index) == [10, 20, 30, 40]
