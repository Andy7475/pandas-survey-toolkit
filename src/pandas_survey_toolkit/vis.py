import textwrap
from typing import List, Tuple

import altair as alt
import pandas as pd


def cluster_heatmap_plot(df: pd.DataFrame, x: str, y: List[str], max_width: int = 75):
    """
    Create a heatmap visualization of Likert scale responses grouped by clusters.

    This function generates an interactive Altair visualization showing the distribution
    of positive and negative responses across different clusters for each question.
    The visualization consists of two parts:
    1. A bar chart showing the number of respondents in each cluster
    2. A heatmap showing the sentiment distribution for each question by cluster

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the clustered data and encoded Likert responses.
        Should include a cluster column and encoded Likert columns.

    x : str
        The name of the column containing cluster IDs (e.g., 'question_cluster_id').

    y : List[str]
        List of column names containing the encoded Likert responses.
        These should typically be columns with values -1, 0, 1 representing
        negative, neutral, and positive responses.

    max_width : int, default=75
        Maximum width for wrapping question labels in the visualization.

    Returns
    -------
    alt.VConcatChart
        An Altair chart object combining a bar chart of cluster sizes and
        a heatmap of sentiment distribution that can be displayed in a Jupyter notebook
        or exported as HTML.

    Notes
    -----
    The function color-codes the heatmap cells based on the percentage of
    positive and negative responses, with green representing positive sentiment,
    red representing negative sentiment, and varying shades for mixed responses.

    The encoded Likert columns (y parameter) should contain values that are encoded as:
    * 1 for positive responses
    * 0 for neutral responses
    * -1 for negative responses

    Examples
    --------
    >>> # Assuming df has been processed with cluster_questions
    >>> likert_columns = [f"likert_encoded_{q}" for q in questions]
    >>> heatmap = cluster_heatmap_plot(df, x="question_cluster_id", y=likert_columns)
    >>> display(heatmap)
    """
    # Convert -1, 0, 1 to percent positive and percent negative
    df_positive = df[y].apply(lambda col: (col == 1).astype(int))
    df_negative = df[y].apply(lambda col: (col == -1).astype(int))

    # Calculate average percent positive and negative for each cluster and question
    heatmap_data_pos = (
        df_positive.groupby(df[x])
        .mean()
        .reset_index()
        .melt(id_vars=x, var_name="question", value_name="percent_positive")
    )
    heatmap_data_neg = (
        df_negative.groupby(df[x])
        .mean()
        .reset_index()
        .melt(id_vars=x, var_name="question", value_name="percent_negative")
    )

    # Merge positive and negative data
    heatmap_data = pd.merge(heatmap_data_pos, heatmap_data_neg, on=[x, "question"])
    heatmap_data["percent_neutral"] = (
        1 - heatmap_data["percent_positive"] - heatmap_data["percent_negative"]
    )

    # Calculate overall positivity for each cluster
    cluster_positivity = (
        heatmap_data.groupby(x)["percent_positive"].mean().sort_values(ascending=False)
    )
    cluster_order = cluster_positivity.index.tolist()

    # Replace underscores with spaces in question labels
    heatmap_data["question"] = (
        heatmap_data["question"].str.replace("_", " ").str.replace("likert encoded", "")
    )

    # Wrap long question labels
    wrapped_labels = [
        textwrap.fill(label, width=max_width)
        for label in heatmap_data["question"].unique()
    ]
    label_to_wrapped = dict(zip(heatmap_data["question"].unique(), wrapped_labels))
    heatmap_data["wrapped_question"] = heatmap_data["question"].map(label_to_wrapped)

    # Define color scale based on percent positive and percent negative
    def get_color(pos: float, neg: float) -> Tuple[str, str]:
        if pos > 0.6:
            return "#1a9641", "white"  # Strong positive (green)
        elif pos > 0.4:
            return "#a6d96a", "black"  # Moderate positive (light green)
        elif pos > neg:
            return "#ffffbf", "black"  # Slightly positive (light yellow)
        elif neg > 0.6:
            return "#d7191c", "white"  # Strong negative (red)
        elif neg > 0.4:
            return "#fdae61", "black"  # Moderate negative (orange)
        elif neg > pos:
            return "#f4a582", "black"  # Slightly negative (light red)
        else:
            return "#f7f7f7", "black"  # Neutral (light gray)

    heatmap_data["background_color"], heatmap_data["text_color"] = zip(
        *heatmap_data.apply(
            lambda row: get_color(row["percent_positive"], row["percent_negative"]),
            axis=1,
        )
    )

    # Calculate chart dimensions
    chart_width = 600
    row_height = 30
    heatmap_height = len(wrapped_labels) * row_height
    bar_chart_height = 100

    # Create heatmap
    heatmap = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:O", title="Cluster ID", sort=cluster_order),
            y=alt.Y("wrapped_question:O", title=None, sort=wrapped_labels),
            color=alt.Color("background_color:N", scale=None),
            tooltip=[
                alt.Tooltip(f"{x}:O", title="Cluster ID"),
                alt.Tooltip("question:O", title="Question"),
                alt.Tooltip("percent_positive:Q", title="% Positive", format=".2%"),
                alt.Tooltip("percent_negative:Q", title="% Negative", format=".2%"),
                alt.Tooltip("percent_neutral:Q", title="% Neutral", format=".2%"),
            ],
        )
        .properties(
            width=chart_width,
            height=heatmap_height,
            title="Cluster Heatmap: Sentiment Distribution",
        )
    )

    # Add text labels to heatmap
    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("percent_positive:Q", format=".0%"),
        color=alt.Color("text_color:N", scale=None),
    )

    # Create bar chart for cluster counts
    cluster_counts = df[x].value_counts().reset_index()
    cluster_counts.columns = [x, "count"]
    cluster_counts[x] = pd.Categorical(
        cluster_counts[x], categories=cluster_order, ordered=True
    )
    cluster_counts = cluster_counts.sort_values(x)

    bar_chart = (
        alt.Chart(cluster_counts)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:O", title="Cluster ID", sort=cluster_order),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip(f"{x}:O", title="Cluster ID"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(width=chart_width, height=bar_chart_height, title="Cluster Sizes")
    )

    # Add text labels to bar chart
    bar_text = bar_chart.mark_text(align="center", baseline="bottom", dy=-5).encode(
        text="count:Q"
    )

    # Combine bar chart and heatmap using vconcat
    combined_chart = (
        alt.vconcat((bar_chart + bar_text), (heatmap + text))
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelLimit=350  # Increase label limit to show full wrapped text
        )
    )

    return combined_chart