import textwrap
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd


def cluster_heatmap_plot(
    df: pd.DataFrame,
    x: str,
    y: List[str],
    max_width: int = 75,
    question_order: Optional[List[str]] = None,
):
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

    question_order : list of str, optional
        Encoded question column names in the order to display them (top to
        bottom), typically the dendrogram order so clustered questions sit
        together. If omitted, it is read from ``df.attrs["question_order"]``
        (set by :func:`pandas_survey_toolkit.nlp.cluster_survey`); if that is
        also absent the questions keep the order of ``y``.

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
    # Order the question rows by their cluster so similar questions sit together.
    # Uses an explicit ``question_order`` if given, else the one stashed on
    # ``df.attrs["question_order"]`` by ``cluster_survey`` / ``cluster_questions``.
    if question_order is None:
        question_order = df.attrs.get("question_order")
    if question_order:
        ordered = [c for c in question_order if c in y]
        y = ordered + [c for c in y if c not in ordered]

    # Work on a plain copy with attrs cleared: heavy objects that helpers may
    # stash in ``df.attrs`` (linkages etc.) otherwise trip up pandas' attrs
    # propagation during the melt/concat below.
    df = df[[x] + list(y)].copy()
    df.attrs = {}

    # Convert encoded responses to percent positive and percent negative.
    # Counting > 0 / < 0 (rather than == 1 / == -1) keeps the 3-point behaviour
    # identical while also supporting the 5-point (+/-2) encoding.
    df_positive = df[y].apply(lambda col: (col > 0).astype(int))
    df_negative = df[y].apply(lambda col: (col < 0).astype(int))

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


def _build_hover_text(
    df: pd.DataFrame,
    hover_cols: Optional[List[str]],
    colour_col: Optional[str],
) -> Optional[List[str]]:
    series_parts = []
    if colour_col:
        series_parts.append(
            df[colour_col].astype(str).apply(lambda v: f"{colour_col}: {v}")
        )
    if hover_cols:
        for col in hover_cols:
            series_parts.append(df[col].astype(str).apply(lambda v, c=col: f"{c}: {v}"))
    if not series_parts:
        return None
    return (
        pd.concat(series_parts, axis=1)
        .apply(lambda row: "\n".join(row), axis=1)
        .tolist()
    )


def _build_marker_color_array(df: pd.DataFrame, colour_col: str) -> np.ndarray:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    unique_vals = df[colour_col].unique()
    n = len(unique_vals)
    cmap = plt.colormaps["tab10" if n <= 10 else "tab20"]
    color_lookup = {
        val: mcolors.to_hex(cmap.colors[i % len(cmap.colors)])
        for i, val in enumerate(unique_vals)
    }
    return np.array([color_lookup[v] for v in df[colour_col]])


def datamap_plot(
    df: pd.DataFrame,
    label_col: str,
    x_col: str,
    y_col: str,
    colour_col: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Create a static DataMapPlot visualisation from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualise.

    label_col : str
        Column name containing cluster/group labels for each point.

    x_col : str
        Column name for the x-axis coordinates (e.g. UMAP dimension 1).

    y_col : str
        Column name for the y-axis coordinates (e.g. UMAP dimension 2).

    colour_col : str or None (optional, default=None)
        Column whose values determine point colours. Each unique value gets a
        distinct colour from the tab10/tab20 palette. When ``None`` datamapplot
        auto-generates colours from the cluster labels.

    title : str or None (optional, default=None)
        Plot title. Passed directly to datamapplot.

    **kwargs
        Additional keyword arguments forwarded to ``datamapplot.create_plot``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes produced by datamapplot.
    """
    import datamapplot

    coords = df[[x_col, y_col]].values
    labels = df[label_col].astype(str).values

    if colour_col is not None:
        kwargs["marker_color_array"] = _build_marker_color_array(df, colour_col)

    if title is not None:
        kwargs["title"] = title

    return datamapplot.create_plot(coords, labels, **kwargs)


def datamap_interactive_plot(
    df: pd.DataFrame,
    label_col: str,
    x_col: str,
    y_col: str,
    hover_cols: Optional[List[str]] = None,
    colour_col: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Create an interactive DataMapPlot visualisation from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualise.

    label_col : str
        Column name containing cluster/group labels for each point.

    x_col : str
        Column name for the x-axis coordinates (e.g. UMAP dimension 1).

    y_col : str
        Column name for the y-axis coordinates (e.g. UMAP dimension 2).

    hover_cols : list of str or None (optional, default=None)
        Columns whose values are shown in the hover tooltip. Multiple columns
        are concatenated with a newline character.

    colour_col : str or None (optional, default=None)
        Column whose values determine point colours. Each unique value gets a
        distinct colour from the tab10/tab20 palette. When ``None`` datamapplot
        auto-generates colours from the cluster labels.

    title : str or None (optional, default=None)
        Plot title. Passed directly to datamapplot.

    **kwargs
        Additional keyword arguments forwarded to
        ``datamapplot.create_interactive_plot``.

    Returns
    -------
    datamapplot.InteractiveFigure
        The interactive figure object; displays inline in Jupyter notebooks.
    """
    import datamapplot

    coords = df[[x_col, y_col]].values
    labels = df[label_col].astype(str).values

    hover_text = _build_hover_text(df, hover_cols, colour_col)
    if hover_text is not None:
        kwargs.setdefault("hover_text", hover_text)

    if colour_col is not None:
        kwargs["marker_color_array"] = _build_marker_color_array(df, colour_col)

    if title is not None:
        kwargs["title"] = title

    return datamapplot.create_interactive_plot(coords, labels, **kwargs)


def plot_respondent_dendrogram(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    color_threshold: Optional[float] = None,
    title: str = "Respondent clustering (cosine distance)",
    ax=None,
    **dendrogram_kwargs,
):
    """Draw the dendrogram produced by ``cluster_respondents_cosine``.

    ``cluster_respondents_cosine`` (directly, or via ``cluster_respondents`` /
    ``cluster_survey``) stores its scipy linkage matrix on
    ``df.attrs["respondent_linkage"]`` and the clustered respondent index on
    ``df.attrs["respondent_linkage_index"]``. This helper renders that linkage as
    a hierarchical dendrogram so you can see how respondents merge and choose a
    cut.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame returned by
        :func:`pandas_survey_toolkit.nlp.cluster_respondents_cosine`.
    label_col : str, optional
        Column to use for leaf labels (e.g. a respondent id column). If None,
        the DataFrame index is used.
    color_threshold : float, optional
        Distance at which to colour the branches (passed straight to scipy's
        ``dendrogram``). Handy to visualise the cut used for clustering.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if None.
    **dendrogram_kwargs
        Extra keyword arguments forwarded to
        :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the dendrogram was drawn on.

    Raises
    ------
    KeyError
        If the DataFrame does not carry the linkage produced by
        ``cluster_respondents_cosine``.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    if "respondent_linkage" not in df.attrs:
        raise KeyError(
            "No linkage found on df.attrs['respondent_linkage']. Run "
            "cluster_respondents_cosine first (and keep the DataFrame it "
            "returns, since df.attrs travels with it)."
        )

    linkage_matrix = np.asarray(df.attrs["respondent_linkage"], dtype=float)
    index = df.attrs.get("respondent_linkage_index")

    if label_col is not None and index is not None:
        labels = df.loc[index, label_col].astype(str).tolist()
    elif index is not None:
        labels = [str(i) for i in index]
    else:
        labels = None

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(linkage_matrix))))

    dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=color_threshold,
        ax=ax,
        **dendrogram_kwargs,
    )
    ax.set_title(title)
    ax.set_ylabel("Cosine distance")
    return ax


def survey_clustermap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    likert_mapping: Optional[dict] = None,
    linkage_method: str = "average",
    label_col: Optional[str] = None,
    max_width: int = 30,
    **clustermap_kwargs,
):
    """Biclustered clustermap of individual respondents x questions (seaborn).

    Rows are respondents and columns are questions, each reordered by
    hierarchical **cosine** clustering with marginal dendrograms, so coherent
    blocks of like-minded respondents and co-answered questions line up. The
    colour scheme is a red -> yellow -> green diverging map (red = disagree,
    green = agree), matching the sentiment colours of
    :func:`cluster_heatmap_plot`.

    This view shows *every* respondent, so it is best for surveys with **few
    respondents**. For thousands of respondents use :func:`cluster_heatmap_plot`,
    which collapses respondents into clusters and adds a cluster-size bar chart.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw survey data (encoding is done internally) or a frame already
        containing ``likert_encoded_*`` columns.
    columns : list of str, optional
        Question columns to plot. If None, ``pattern`` is used.
    pattern : str, optional
        Regex to match question columns (used if ``columns`` is None).
    likert_mapping : dict, optional
        Custom Likert mapping (see :func:`pandas_survey_toolkit.nlp.encode_likert`).
    linkage_method : str, optional
        scipy linkage method for both axes. Default "average".
    label_col : str, optional
        Column to label the respondent (row) axis with. Defaults to the index.
    max_width : int, optional
        Wrap width for question (column) labels.
    **clustermap_kwargs
        Extra keyword arguments forwarded to :func:`seaborn.clustermap`.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The clustermap grid (``.fig`` for the figure).

    Raises
    ------
    ValueError
        If fewer than two respondents or two questions have a non-neutral answer.
    """
    import seaborn as sns

    import pandas_survey_toolkit.nlp  # noqa: F401  registers encode_likert
    from pandas_survey_toolkit.nlp import _select_likert_columns

    columns = _select_likert_columns(df, columns, pattern)
    encoded_columns = [f"likert_encoded_{c}" for c in columns]
    if not all(c in df.columns for c in encoded_columns):
        df = df.encode_likert(columns, custom_mapping=likert_mapping, debug=False)

    data = df[encoded_columns].astype(float).copy()
    data.columns = [
        textwrap.fill(
            c.replace("likert_encoded_", "").replace("_", " "), width=max_width
        )
        for c in encoded_columns
    ]
    if label_col is not None and label_col in df.columns:
        data.index = df[label_col].astype(str).values

    # Cosine distance is undefined for an all-neutral (zero) vector, so drop
    # all-zero rows/cols after filling gaps with 0 (neutral).
    data = data.dropna(how="all").fillna(0.0)
    data = data.loc[(data != 0).any(axis=1), (data != 0).any(axis=0)]
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(
            "Need at least two respondents and two questions with a non-neutral "
            "answer to draw a clustermap."
        )

    clustermap_kwargs.setdefault("metric", "cosine")
    clustermap_kwargs.setdefault("method", linkage_method)
    clustermap_kwargs.setdefault("cmap", "RdYlGn")
    clustermap_kwargs.setdefault("center", 0)
    clustermap_kwargs.setdefault("vmin", -1)
    clustermap_kwargs.setdefault("vmax", 1)
    clustermap_kwargs.setdefault("cbar_kws", {"label": "sentiment"})

    return sns.clustermap(data, **clustermap_kwargs)
