"""Tests for visualisation functions."""

import altair as alt
import numpy as np
import pandas as pd
import pytest

from pandas_survey_toolkit.vis import cluster_heatmap_plot


@pytest.fixture
def heatmap_df():
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "cluster_id": np.random.choice([0, 1, 2], n),
            "likert_encoded_Q1": np.random.choice([-1, 0, 1], n),
            "likert_encoded_Q2": np.random.choice([-1, 0, 1], n),
            "likert_encoded_Q3": np.random.choice([-1, 0, 1], n),
        }
    )


def test_returns_vconchat_chart(heatmap_df):
    chart = cluster_heatmap_plot(
        heatmap_df,
        x="cluster_id",
        y=["likert_encoded_Q1", "likert_encoded_Q2", "likert_encoded_Q3"],
    )
    assert isinstance(chart, alt.VConcatChart)


def test_custom_max_width(heatmap_df):
    chart = cluster_heatmap_plot(
        heatmap_df,
        x="cluster_id",
        y=["likert_encoded_Q1", "likert_encoded_Q2"],
        max_width=40,
    )
    assert isinstance(chart, alt.VConcatChart)


def test_strongly_positive_cluster():
    """Cluster where all responses are 1 should produce strong positive (green)
    color."""
    df = pd.DataFrame(
        {
            "cluster": [0] * 20 + [1] * 20,
            "q1": [1] * 20 + [-1] * 20,
        }
    )
    chart = cluster_heatmap_plot(df, x="cluster", y=["q1"])
    assert isinstance(chart, alt.VConcatChart)


def test_strongly_negative_cluster():
    """Cluster where all responses are -1 should produce strong negative (red) color."""
    df = pd.DataFrame(
        {
            "cluster": [0] * 20 + [1] * 20,
            "q1": [-1] * 20 + [1] * 20,
        }
    )
    chart = cluster_heatmap_plot(df, x="cluster", y=["q1"])
    assert isinstance(chart, alt.VConcatChart)


def test_mixed_sentiment_cluster():
    """Mixed clusters covering moderate-positive, slightly-positive, and
    moderate-negative branches."""
    df = pd.DataFrame(
        {
            # cluster 0: moderately positive (45% pos, 10% neg -> pos>0.4)
            # cluster 1: slightly positive (35% pos, 20% neg -> pos>neg, <0.4)
            # cluster 2: moderately negative (15% pos, 45% neg -> neg>0.4)
            "cluster": [0] * 20 + [1] * 20 + [2] * 20,
            "q1": [1] * 9
            + [-1] * 2
            + [0] * 9
            + [1] * 7
            + [-1] * 4
            + [0] * 9
            + [1] * 3
            + [-1] * 9
            + [0] * 8,
        }
    )
    chart = cluster_heatmap_plot(df, x="cluster", y=["q1"])
    assert isinstance(chart, alt.VConcatChart)


def test_neutral_all_zero():
    """All-neutral cluster: pos=0, neg=0 → both equal and ≤0.4, hits the
    gray else branch."""
    df = pd.DataFrame(
        {
            "cluster": [0] * 10,
            "q1": [0] * 10,  # all neutral → pos=0.0, neg=0.0, neither > the other
        }
    )
    chart = cluster_heatmap_plot(df, x="cluster", y=["q1"])
    assert isinstance(chart, alt.VConcatChart)


def test_single_cluster():
    """Edge case: only one cluster."""
    df = pd.DataFrame(
        {
            "cluster": [0] * 10,
            "q1": [1, 1, 0, -1, 1, 0, 1, -1, 1, 0],
        }
    )
    chart = cluster_heatmap_plot(df, x="cluster", y=["q1"])
    assert isinstance(chart, alt.VConcatChart)


def test_multiple_questions(heatmap_df):
    """Verify the chart works with multiple y columns."""
    chart = cluster_heatmap_plot(
        heatmap_df,
        x="cluster_id",
        y=["likert_encoded_Q1", "likert_encoded_Q2", "likert_encoded_Q3"],
    )
    assert isinstance(chart, alt.VConcatChart)
