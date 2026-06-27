"""Tests for visualisation functions."""

import re

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from pandas_survey_toolkit.vis import (  # noqa: E402
    _build_hover_text,
    _build_marker_color_array,
    cluster_heatmap_plot,
    datamap_interactive_plot,
    datamap_plot,
)


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEX_COLOR_RE = re.compile(r"^#[0-9a-f]{6}$", re.IGNORECASE)


@pytest.fixture
def datamap_df():
    np.random.seed(0)
    n = 30
    return pd.DataFrame(
        {
            "label": np.tile(["Cluster A", "Cluster B", "Cluster C"], n // 3),
            "x": np.random.randn(n),
            "y": np.random.randn(n),
            "response": [f"response text {i}" for i in range(n)],
            "note": [f"note {i}" for i in range(n)],
            "gender": np.tile(["M", "F"], n // 2),
        }
    )


# ---------------------------------------------------------------------------
# _build_hover_text
# ---------------------------------------------------------------------------


def test_hover_text_no_cols_no_colour(datamap_df):
    assert _build_hover_text(datamap_df, hover_cols=None, colour_col=None) is None


def test_hover_text_hover_cols_only(datamap_df):
    result = _build_hover_text(datamap_df, hover_cols=["response"], colour_col=None)
    assert result is not None
    assert len(result) == len(datamap_df)
    assert result[0] == "response text 0"


def test_hover_text_multiple_hover_cols(datamap_df):
    result = _build_hover_text(
        datamap_df, hover_cols=["response", "note"], colour_col=None
    )
    assert result is not None
    # Each entry should join the two columns with a newline
    assert result[0] == "response text 0\nnote 0"


def test_hover_text_colour_col_only(datamap_df):
    result = _build_hover_text(datamap_df, hover_cols=None, colour_col="gender")
    assert result is not None
    assert len(result) == len(datamap_df)
    assert result[0] == "gender: M"
    assert result[1] == "gender: F"


def test_hover_text_colour_col_appears_first(datamap_df):
    result = _build_hover_text(datamap_df, hover_cols=["response"], colour_col="gender")
    assert result is not None
    first_line, second_line = result[0].split("\n")
    assert first_line == "gender: M"
    assert second_line == "response text 0"


def test_hover_text_colour_and_multiple_hover_cols(datamap_df):
    result = _build_hover_text(
        datamap_df, hover_cols=["response", "note"], colour_col="gender"
    )
    assert result is not None
    parts = result[0].split("\n")
    assert len(parts) == 3
    assert parts[0] == "gender: M"
    assert parts[1] == "response text 0"
    assert parts[2] == "note 0"


# ---------------------------------------------------------------------------
# _build_marker_color_array
# ---------------------------------------------------------------------------


def test_color_array_length(datamap_df):
    colors = _build_marker_color_array(datamap_df, "gender")
    assert len(colors) == len(datamap_df)


def test_color_array_hex_format(datamap_df):
    colors = _build_marker_color_array(datamap_df, "label")
    for color in colors:
        assert HEX_COLOR_RE.match(color), f"{color!r} is not a valid hex color"


def test_color_array_same_value_same_color(datamap_df):
    colors = _build_marker_color_array(datamap_df, "gender")
    color_m = colors[datamap_df["gender"] == "M"][0]
    assert all(c == color_m for c in colors[datamap_df["gender"] == "M"])


def test_color_array_different_values_different_colors(datamap_df):
    colors = _build_marker_color_array(datamap_df, "gender")
    color_m = colors[datamap_df["gender"] == "M"][0]
    color_f = colors[datamap_df["gender"] == "F"][0]
    assert color_m != color_f


def test_color_array_more_than_ten_unique_values():
    df = pd.DataFrame({"cat": [str(i) for i in range(15)]})
    colors = _build_marker_color_array(df, "cat")
    assert len(colors) == 15
    assert all(HEX_COLOR_RE.match(c) for c in colors)
    # All 15 unique values should each get a distinct color
    assert len(set(colors)) == 15


# ---------------------------------------------------------------------------
# datamap_plot (static)
# ---------------------------------------------------------------------------


def test_datamap_plot_returns_tuple(datamap_df):
    import matplotlib.figure

    result = datamap_plot(datamap_df, "label", "x", "y")
    assert isinstance(result, tuple)
    assert isinstance(result[0], matplotlib.figure.Figure)


def test_datamap_plot_with_colour_col(datamap_df):
    import matplotlib.figure

    result = datamap_plot(datamap_df, "label", "x", "y", colour_col="gender")
    assert isinstance(result[0], matplotlib.figure.Figure)


def test_datamap_plot_with_title(datamap_df):
    import matplotlib.figure

    result = datamap_plot(datamap_df, "label", "x", "y", title="My Map")
    assert isinstance(result[0], matplotlib.figure.Figure)


def test_datamap_plot_colour_and_title(datamap_df):
    import matplotlib.figure

    result = datamap_plot(
        datamap_df, "label", "x", "y", colour_col="gender", title="Test Plot"
    )
    assert isinstance(result[0], matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# datamap_interactive_plot
# ---------------------------------------------------------------------------


def test_datamap_interactive_plot_returns_interactive_figure(datamap_df):
    from datamapplot.interactive_rendering import InteractiveFigure

    result = datamap_interactive_plot(datamap_df, "label", "x", "y")
    assert isinstance(result, InteractiveFigure)


def test_datamap_interactive_plot_with_colour_col(datamap_df):
    from datamapplot.interactive_rendering import InteractiveFigure

    result = datamap_interactive_plot(
        datamap_df, "label", "x", "y", colour_col="gender"
    )
    assert isinstance(result, InteractiveFigure)


def test_datamap_interactive_plot_with_hover_cols(datamap_df):
    from datamapplot.interactive_rendering import InteractiveFigure

    result = datamap_interactive_plot(
        datamap_df, "label", "x", "y", hover_cols=["response", "note"]
    )
    assert isinstance(result, InteractiveFigure)


def test_datamap_interactive_plot_all_options(datamap_df):
    from datamapplot.interactive_rendering import InteractiveFigure

    result = datamap_interactive_plot(
        datamap_df,
        "label",
        "x",
        "y",
        hover_cols=["response"],
        colour_col="gender",
        title="Test Interactive",
    )
    assert isinstance(result, InteractiveFigure)
