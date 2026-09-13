import re
import warnings
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from gensim.parsing.preprocessing import (
    remove_stopwords as gensim_remove_stopwords,
)
from gensim.parsing.preprocessing import (
    strip_multiple_whitespaces,
    strip_numeric,
    strip_tags,
)
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.special import softmax

# NOTE: the heavy free-text NLP libraries (spacy, sentence-transformers,
# transformers, and their torch dependency) are the optional ``nlp`` extra. They
# are imported lazily inside the functions that need them so that the Likert
# clustering and text-preprocessing functions (encode_likert, cluster_questions,
# cluster_respondents, cluster_survey, preprocess_text, clean_survey_columns,
# ...) work with a torch-free install. gensim is lightweight and stays core.
#     pip install pandas-survey-toolkit          # clustering + preprocessing
#     pip install "pandas-survey-toolkit[nlp]"   # + free-text comment embeddings
# Importing analytics registers the fit_umap / fit_cluster_hdbscan dataframe
# methods that cluster_respondents relies on.
from pandas_survey_toolkit import analytics  # noqa: F401
from pandas_survey_toolkit.utils import (
    apply_vectorizer,
    combine_results,
    create_masked_df,
)


def _require_nlp_extra(feature):
    """Raise a helpful error when the optional ``nlp`` extra is not installed."""
    raise ImportError(
        f"{feature} needs the optional 'nlp' extra (spacy, gensim, "
        "sentence-transformers, transformers). Install it with:\n"
        '    pip install "pandas-survey-toolkit[nlp]"'
    )


def _select_likert_columns(df, columns, pattern):
    """Resolve the list of Likert columns from an explicit list or a regex."""
    if columns is None and pattern is None:
        raise ValueError("Either 'columns' or 'pattern' must be provided.")
    if columns is None:
        columns = df.filter(regex=pattern).columns.tolist()
    return columns


def _select_clustering_method(n_respondents, size_threshold=1000):
    """Pick a respondent-clustering method from the survey size.

    Cosine + hierarchical clustering builds an ``n x n`` distance matrix - great
    for small/medium surveys but ``O(n^2)``; UMAP + HDBSCAN scales to very large
    surveys (and needs many respondents to be reliable). Returns ``"cosine"`` at
    or below ``size_threshold`` respondents, otherwise ``"umap"``.
    """
    return "cosine" if n_respondents <= size_threshold else "umap"


def _check_for_nan(df, columns, nan_strategy="fill", fill_value=0):
    """Handle missing Likert answers before clustering.

    Always warns if any NaN is found among ``columns`` (regardless of
    strategy): a NaN here may just mean an unanswered question, but it can also
    mean :func:`encode_likert` silently failed to map a raw response, so the
    caller should notice either way.

    ``nan_strategy="fill"`` (default): missing answers are additionally treated
    as neutral and filled with ``fill_value``. ``nan_strategy="ignore"``: rows
    with any missing answer among ``columns`` are dropped entirely (reuses
    :func:`pandas_survey_toolkit.utils.create_masked_df`).

    Raises
    ------
    ValueError
        If ``nan_strategy`` is not "fill" or "ignore".
    """
    if nan_strategy not in ("fill", "ignore"):
        raise ValueError(
            f"nan_strategy must be 'fill' or 'ignore', got {nan_strategy!r}."
        )

    n_missing = int(df[columns].isna().sum().sum())
    if n_missing == 0:
        return df

    warnings.warn(
        f"{n_missing} missing Likert answer(s) found (unanswered question, or "
        "encode_likert failed to map a raw response - check its warnings too).",
        stacklevel=3,
    )

    if nan_strategy == "ignore":
        masked_df, mask = create_masked_df(df, columns)
        warnings.warn(
            f"{int((~mask).sum())} row(s) had at least one missing Likert answer "
            "and were excluded from clustering (nan_strategy='ignore').",
            stacklevel=3,
        )
        return masked_df

    warnings.warn(
        f"{n_missing} missing Likert answer(s) were treated as neutral "
        f"(filled with {fill_value}); pass nan_strategy='ignore' to exclude "
        "affected respondents instead.",
        stacklevel=3,
    )
    df = df.copy()
    # The encoded columns are object-dtype (int/pd.NA mix); convert to float64
    # first (pd.to_numeric handles pd.NA, unlike a direct astype) so fillna
    # doesn't trigger pandas' object-dtype downcasting warning.
    df[columns] = df[columns].apply(pd.to_numeric).fillna(fill_value)
    return df


def _check_encoded_range(df, columns, likert_mapping):
    """Warn if encoded Likert data contains values outside the expected scale.

    A stray value that never went through :func:`encode_likert` (e.g. a raw,
    un-encoded column mistakenly passed as ``columns``) would silently distort
    cosine norms without erroring, so this is a cheap sanity check rather than
    a hard invariant. The expected scale is ``{-1, 0, 1}`` for the default
    mapping, or the set of values used by a ``custom_mapping``/``likert_mapping``.
    """
    valid_values = (
        {-1, 0, 1} if likert_mapping is None else set(likert_mapping.values())
    )
    ok = df[columns].isin(valid_values) | df[columns].isna()
    if not ok.all().all():
        warnings.warn(
            "Encoded Likert data contains values outside the expected "
            f"{sorted(valid_values)} (besides NaN); check that 'columns' points "
            "at already-encoded Likert data and not something else.",
            stacklevel=3,
        )


def _cosine_cluster(
    data,
    *,
    object_label,
    linkage_method="average",
    n_clusters=None,
    distance_threshold=None,
):
    """Hierarchically cluster the *columns* of ``data`` by cosine distance.

    Shared engine behind :func:`cluster_respondents_cosine` and
    :func:`cluster_questions`. ``data`` is a numeric DataFrame whose columns are
    the objects to cluster (respondents or questions) and whose rows are their
    encoded answers. Callers are expected to have already resolved missing
    values (see :func:`_check_for_nan`) - ``data`` should not contain NaN.

    Cosine distance keeps the *direction* of a response vector, so respondents
    who all agree and respondents who all disagree point opposite ways and
    separate cleanly - unlike a correlation, which subtracts each respondent's
    mean and cannot even be defined for a flat (zero-variance) "always agree"
    response. The one case cosine cannot place is a **zero vector** (all-neutral /
    all-zero answers): it has no direction, so it is left unclustered (``-1``).

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric matrix; its columns are the response vectors to cluster.
    object_label : str
        Noun used in warnings ("respondent" or "question").
    linkage_method : str, optional
        scipy linkage method. Must be one of "single", "complete", "average" or
        "weighted" - "ward"/"centroid"/"median" assume Euclidean distances and
        produce meaningless merges on a cosine distance matrix.
    n_clusters, distance_threshold
        See :func:`cluster_respondents_cosine`. When neither is given, the
        dendrogram is cut at cosine distance 1.0 (response directions that are
        orthogonal or more get split into different clusters).

    Returns
    -------
    labels : pandas.Series
        Integer cluster id per column of ``data`` (index == ``data.columns``);
        ``-1`` marks all-neutral columns that could not be clustered.
    linkage_matrix : numpy.ndarray or None
        scipy linkage over the clustered columns (ordered as ``clustered``), or
        ``None`` if fewer than two columns could be clustered.
    clustered : list
        Clustered column names in original order (matches the linkage leaves).
    order : list
        Clustered column names in dendrogram-leaf order (adjacent == similar).

    Notes
    -----
    With 3-point (-1/0/1) data, many respondents/questions can be exactly
    identical, producing tied (zero) distances. Average linkage still clusters
    them correctly, but the *leaf order* (``order``) among tied entries is not
    guaranteed stable across scipy versions - fine unless you rely on it for
    regression tests.
    """
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError("Provide at most one of 'n_clusters' or 'distance_threshold'.")

    valid_linkage = {"single", "complete", "average", "weighted"}
    if linkage_method not in valid_linkage:
        raise ValueError(
            f"linkage_method must be one of {sorted(valid_linkage)} for cosine "
            f"distances, got {linkage_method!r}. 'ward'/'centroid'/'median' assume "
            "Euclidean distances and produce meaningless results here."
        )

    labels = pd.Series(-1, index=data.columns, dtype=int)

    # Each column is a response vector (callers guarantee no NaN reaches here).
    vectors = data.to_numpy(dtype=float).T  # (n_objects, n_obs)
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 0  # a flat all-neutral (zero) vector has no direction -> -1
    clustered = list(data.columns[keep])
    n_degenerate = len(data.columns) - len(clustered)
    if n_degenerate:
        warnings.warn(
            f"{n_degenerate} {object_label}(s) gave an all-neutral (zero) response;"
            " cosine distance is undefined for them so they are left unclustered"
            " (cluster -1).",
            stacklevel=3,
        )

    if len(clustered) < 2:
        warnings.warn(
            f"Fewer than two {object_label}s could be clustered; no clusters formed.",
            stacklevel=3,
        )
        return labels, None, clustered, clustered

    # Condensed pairwise cosine distances (1 - cosine similarity), in [0, 2].
    condensed = np.clip(
        np.nan_to_num(pdist(vectors[keep], metric="cosine"), nan=1.0), 0.0, 2.0
    )
    linkage_matrix = linkage(condensed, method=linkage_method)

    if n_clusters is not None:
        lab = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    else:
        threshold = 1.0 if distance_threshold is None else distance_threshold
        lab = fcluster(linkage_matrix, t=threshold, criterion="distance")

    labels.loc[clustered] = lab.astype(int)
    order = [clustered[i] for i in leaves_list(linkage_matrix)]

    return labels, linkage_matrix, clustered, order


@pf.register_dataframe_method
def cluster_respondents(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    method="auto",
    size_threshold=1000,
    nan_strategy="fill",
    fill_value=0,
    output_columns=("respondent_cluster_id", "respondent_cluster_probability"),
    debug=False,
    **kwargs,
):
    """Cluster respondents, choosing the method by survey size.

    Dispatches to one of two engines and forwards any extra keyword arguments to
    the chosen one:

    - ``method="cosine"`` -> :func:`cluster_respondents_cosine` (hierarchical
      clustering on cosine distance; best for small/medium surveys, ``O(n^2)``).
    - ``method="umap"`` -> :func:`cluster_respondents_umap` (UMAP + HDBSCAN;
      scales to very large surveys, needs many respondents).
    - ``method="auto"`` (default) -> :func:`_select_clustering_method` picks
      cosine at or below ``size_threshold`` respondents, otherwise umap.

    An explicit choice that is a poor fit for the survey size emits a warning.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list, optional
        Question column names to use. If None, ``pattern`` is used.
    pattern : str, optional
        Regex pattern to match column names. Used if ``columns`` is None.
    likert_mapping : dict, optional
        Custom mapping for Likert responses (see :func:`encode_likert`).
    method : {"auto", "cosine", "umap"}, optional
        Which clustering engine to use. Default "auto".
    size_threshold : int, optional
        Respondent count above which "auto" switches to UMAP. Default 1000.
    nan_strategy : {"fill", "ignore"}, optional
        How to handle missing Likert answers, forwarded to whichever engine is
        chosen. "fill" (default) treats missing answers as neutral (see
        ``fill_value``); "ignore" drops respondents with any missing answer.
        See :func:`cluster_respondents_cosine` / :func:`cluster_respondents_umap`.
    fill_value : float, optional
        Value used for missing answers when ``nan_strategy="fill"``. Default 0.
    output_columns : tuple, optional
        (cluster id, cluster probability) column names. The cosine method only
        writes the cluster-id column. Default
        ("respondent_cluster_id", "respondent_cluster_probability").
    debug : bool, optional
        Forwarded to :func:`encode_likert`. Default False.
    **kwargs
        Forwarded to the chosen method (e.g. ``n_clusters`` /
        ``distance_threshold`` for cosine; ``umap_n_neighbors`` etc. for umap).

    Returns
    -------
    pandas.DataFrame
        With a ``respondent_cluster_id`` column (and, for umap, a probability
        column and UMAP coordinates).

    Raises
    ------
    ValueError
        For an unknown ``method`` or if neither 'columns' nor 'pattern' is given.
    """
    if method not in ("auto", "cosine", "umap"):
        raise ValueError(f"method must be 'auto', 'cosine' or 'umap', got {method!r}.")

    columns = _select_likert_columns(df, columns, pattern)
    # With nan_strategy="fill" every respondent ends up clustered, so size
    # selection should count all of them; "ignore" still drops incomplete ones.
    if nan_strategy == "fill":
        n_respondents = int(len(df))
    else:
        n_respondents = int(df[columns].notna().all(axis=1).sum())

    chosen = method
    if method == "auto":
        chosen = _select_clustering_method(n_respondents, size_threshold=size_threshold)
    elif method == "cosine" and n_respondents > size_threshold:
        warnings.warn(
            f"method='cosine' builds a {n_respondents}x{n_respondents} distance "
            "matrix, which is expensive at this size; consider 'umap' or 'auto'.",
            stacklevel=2,
        )
    elif method == "umap" and n_respondents < 50:
        warnings.warn(
            f"method='umap' is unreliable with only {n_respondents} respondents; "
            "consider 'cosine' or 'auto'.",
            stacklevel=2,
        )

    if chosen == "cosine":
        return df.cluster_respondents_cosine(
            columns=columns,
            likert_mapping=likert_mapping,
            nan_strategy=nan_strategy,
            fill_value=fill_value,
            output_column=output_columns[0],
            debug=debug,
            **kwargs,
        )
    return df.cluster_respondents_umap(
        columns=columns,
        likert_mapping=likert_mapping,
        nan_strategy=nan_strategy,
        fill_value=fill_value,
        output_columns=output_columns,
        debug=debug,
        **kwargs,
    )


@pf.register_dataframe_method
def cluster_respondents_umap(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    hdbscan_min_cluster_size=20,
    hdbscan_min_samples=None,
    cluster_selection_epsilon=0.4,
    nan_strategy="fill",
    fill_value=0,
    output_columns=("respondent_cluster_id", "respondent_cluster_probability"),
    debug=False,
):
    """Cluster respondents by their Likert responses with UMAP + HDBSCAN.

    Each respondent (row) becomes a vector of encoded answers, embedded with UMAP
    (cosine metric) and grouped with HDBSCAN. UMAP embeds one point *per
    respondent*, so this needs a reasonable number of respondents; for small
    surveys prefer :func:`cluster_respondents_cosine`.
    :func:`cluster_respondents` chooses between the two automatically. See
    ``docs/source/clustering_methods_comparison.md`` for the maths.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list, optional
        Question column names to use. If None, ``pattern`` is used.
    pattern : str, optional
        Regex pattern to match column names. Used if ``columns`` is None.
    likert_mapping : dict, optional
        Custom mapping for Likert responses (see :func:`encode_likert`).
    umap_n_neighbors : int, optional
        The size of local neighborhood for UMAP. Default is 15.
    umap_min_dist : float, optional
        The minimum distance between points in UMAP. Default is 0.1.
    hdbscan_min_cluster_size : int, optional
        The minimum size of clusters for HDBSCAN. Default is 20.
    hdbscan_min_samples : int, optional
        The number of samples in a neighborhood for a core point in HDBSCAN.
        Default is None.
    cluster_selection_epsilon : float, optional
        A distance threshold. Clusters below this value will be merged. Default
        is 0.4. Higher epsilon means fewer, larger clusters.
    nan_strategy : {"fill", "ignore"}, optional
        How to handle missing Likert answers. "fill" (default) treats missing
        answers as neutral, filled with ``fill_value``, so every respondent gets
        embedded and clustered. "ignore" drops respondents with any missing
        answer before embedding - they end up with NaN coordinates/cluster id,
        matching this function's behavior prior to the ``nan_strategy`` option.
    fill_value : float, optional
        Value used for missing answers when ``nan_strategy="fill"``. Default 0.
    output_columns : tuple, optional
        Names for the (cluster id, cluster probability) output columns. Default
        is ("respondent_cluster_id", "respondent_cluster_probability").
    debug : bool, optional
        Forwarded to :func:`encode_likert`. Default False.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with encoded Likert responses, UMAP coordinates, and
        cluster IDs.

    Raises
    ------
    ValueError
        If neither 'columns' nor 'pattern' is provided, or ``nan_strategy`` is
        invalid.
    """
    columns = _select_likert_columns(df, columns, pattern)

    # Encode Likert scales
    df = df.encode_likert(columns, custom_mapping=likert_mapping, debug=debug)
    encoded_columns = [f"likert_encoded_{col}" for col in columns]

    if nan_strategy == "fill":
        # Shape-preserving: safe to swap in directly so fit_umap sees no NaN
        # left to mask (every respondent gets embedded and clustered).
        df = _check_for_nan(
            df, encoded_columns, nan_strategy=nan_strategy, fill_value=fill_value
        )
    else:
        # "ignore": just raise the warnings here. fit_umap/fit_cluster_hdbscan
        # already exclude incomplete respondents via create_masked_df and
        # preserve the full row count (NaN result for excluded rows), so there
        # is nothing to swap into ``df``.
        _check_for_nan(
            df, encoded_columns, nan_strategy=nan_strategy, fill_value=fill_value
        )

    # Guard against surveys that are too small for density-based clustering.
    # HDBSCAN raises if min_samples exceeds the number of points, and cannot
    # form a cluster of ``min_cluster_size`` if there are fewer respondents than
    # that. Shrink the parameters (with a warning) so the call degrades to a
    # trivial result instead of erroring, and point users at the cosine method
    # which is designed for this regime.
    n_respondents = int(df[encoded_columns].notna().all(axis=1).sum())
    effective_min_cluster_size = hdbscan_min_cluster_size
    effective_min_samples = hdbscan_min_samples
    if hdbscan_min_cluster_size > n_respondents:
        effective_min_cluster_size = max(2, n_respondents)
        if effective_min_samples is not None:
            effective_min_samples = min(effective_min_samples, n_respondents)
        warnings.warn(
            f"hdbscan_min_cluster_size ({hdbscan_min_cluster_size}) exceeds the "
            f"number of complete respondents ({n_respondents}); reducing it to "
            f"{effective_min_cluster_size}. UMAP+HDBSCAN is unreliable with so few "
            "respondents - consider cluster_respondents_cosine, which is designed "
            "for surveys with few respondents and many questions.",
            stacklevel=2,
        )

    # Apply UMAP
    df = df.fit_umap(
        input_columns=encoded_columns,
        output_columns=["likert_umap_x", "likert_umap_y"],
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
    )

    # Apply HDBSCAN
    df = df.fit_cluster_hdbscan(
        input_columns=["likert_umap_x", "likert_umap_y"],
        output_columns=list(output_columns),
        min_cluster_size=effective_min_cluster_size,
        min_samples=effective_min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    return df


@pf.register_dataframe_method
def cluster_questions(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    linkage_method="average",
    n_clusters=None,
    distance_threshold=None,
    nan_strategy="fill",
    fill_value=0,
    debug=False,
):
    """Cluster *questions* by how respondents co-answer them.

    This groups questions that get similar response patterns across respondents,
    purely from the data (no NLP on the question text). It is the mirror of
    :func:`cluster_respondents_cosine`: instead of comparing respondents across
    the questions, it compares the questions across the respondents, using
    **cosine distance** and hierarchical clustering. Cosine keeps the direction
    of each question's response vector, so a question everyone agrees with and a
    question everyone disagrees with point opposite ways and separate cleanly.

    .. note::
        This is a breaking change from v1.x, where ``cluster_questions`` was a
        deprecated alias that actually clustered *respondents*. It now clusters
        questions. Use :func:`cluster_respondents` to cluster respondents.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list, optional
        Question column names to cluster. If None, ``pattern`` is used.
    pattern : str, optional
        Regex pattern to match column names. Used if ``columns`` is None.
    likert_mapping : dict, optional
        Custom mapping for Likert responses (see :func:`encode_likert`).
    linkage_method : str, optional
        scipy linkage method ("average", "complete", ...). Default "average".
    n_clusters : int, optional
        Cut the dendrogram to exactly this many clusters. Mutually exclusive
        with ``distance_threshold``.
    distance_threshold : float, optional
        Cosine cophenetic-distance cut (defaults to 1.0 when neither is given,
        i.e. split questions whose response directions are orthogonal or more).
    nan_strategy : {"fill", "ignore"}, optional
        How to handle missing Likert answers. "fill" (default) treats a missing
        answer as neutral, filled with ``fill_value``, so every respondent's
        answer still contributes to each question's response vector. "ignore"
        drops any respondent with a missing answer among ``columns`` before
        clustering questions.
    fill_value : float, optional
        Value used for missing answers when ``nan_strategy="fill"``. Default 0.
    debug : bool, optional
        Forwarded to :func:`encode_likert`. Default False.

    Returns
    -------
    pandas.Series
        A Series **indexed by the original question column names**, giving an
        integer cluster id per question (``-1`` for a question everyone answered
        *neutral*, a zero vector with no direction). The scipy linkage and the
        dendrogram-leaf orderings are attached on ``.attrs``:
        ``.attrs["linkage"]``, ``.attrs["question_order"]`` (original names,
        similar questions adjacent) and ``.attrs["encoded_order"]`` (the
        ``likert_encoded_*`` column names in the same order).

    Examples
    --------
    Use it standalone and save the mapping for other analysis::

        question_clusters = df.cluster_questions(columns=likert_cols)
        # -> pandas.Series indexed by question name, values are cluster ids
        question_clusters.to_csv("question_clusters.csv")

        # questions grouped, in dendrogram order:
        for q in question_clusters.attrs["question_order"]:
            print(question_clusters[q], q)

    Raises
    ------
    ValueError
        If neither 'columns' nor 'pattern' is given, if both ``n_clusters`` and
        ``distance_threshold`` are given, or if ``nan_strategy`` is invalid.
    """
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError("Provide at most one of 'n_clusters' or 'distance_threshold'.")

    columns = _select_likert_columns(df, columns, pattern)
    encoded = df.encode_likert(columns, custom_mapping=likert_mapping, debug=debug)
    encoded_columns = [f"likert_encoded_{col}" for col in columns]

    _check_encoded_range(encoded, encoded_columns, likert_mapping)
    checked = _check_for_nan(
        encoded, encoded_columns, nan_strategy=nan_strategy, fill_value=fill_value
    )

    # rows = respondents, columns = questions -> clustering the columns.
    responses = checked[encoded_columns].astype(float)

    labels, linkage_matrix, _clustered, order = _cosine_cluster(
        responses,
        object_label="question",
        linkage_method=linkage_method,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
    )

    # Report against the original (un-prefixed) question names.
    enc_to_orig = dict(zip(encoded_columns, columns))
    result = pd.Series(
        labels.to_numpy(),
        index=[enc_to_orig[c] for c in labels.index],
        name="question_cluster_id",
        dtype=int,
    )
    result.attrs["linkage"] = linkage_matrix
    result.attrs["encoded_order"] = order
    result.attrs["question_order"] = [enc_to_orig[c] for c in order]
    result.attrs["encoded_columns"] = encoded_columns
    return result


@pf.register_dataframe_method
def cluster_respondents_cosine(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    linkage_method="average",
    n_clusters=None,
    distance_threshold=None,
    nan_strategy="fill",
    fill_value=0,
    output_column="respondent_cluster_id",
    debug=False,
):
    """Cluster respondents using cosine distance + hierarchical clustering.

    The recommended approach for **small/medium surveys**. Each respondent is a
    vector of their encoded answers; respondents are grouped by the **cosine
    distance** between those vectors and agglomerative (dendrogram) clustering.

    Cosine compares the *direction* of the answer vectors, which is exactly what
    you want for opinion data: respondents who agree with everything point one
    way and respondents who disagree with everything point the opposite way, so
    they separate cleanly - whereas a correlation subtracts each respondent's
    mean and cannot even be defined for a flat "always agree" response. A
    respondent who answers **everything neutral** is a zero vector with no
    direction and is left unclustered (``-1``). See
    ``docs/source/clustering_methods_comparison.md`` for the maths.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list, optional
        List of question column names to use. If None, ``pattern`` is used.
    pattern : str, optional
        Regex pattern to match column names. Used if ``columns`` is None.
    likert_mapping : dict, optional
        Custom mapping for Likert responses (see :func:`encode_likert`).
    linkage_method : str, optional
        Linkage method for :func:`scipy.cluster.hierarchy.linkage`
        ("average", "complete", "single", ...). Default is "average".
    n_clusters : int, optional
        If given, cut the dendrogram to produce exactly this many clusters
        (``criterion="maxclust"``). Mutually exclusive with ``distance_threshold``.
    distance_threshold : float, optional
        If given, cut the dendrogram at this cosine cophenetic distance
        (``criterion="distance"``): respondents merge while their distance stays
        below the threshold. Higher -> fewer, larger clusters (more disagreement
        tolerated); lower -> more, smaller clusters. If neither ``n_clusters`` nor
        ``distance_threshold`` is provided, a threshold of 1.0 is used (cosine
        distance 1 = orthogonal answer directions; for a 10-question +/-1 survey
        that is roughly "disagree on 5+ questions").

        Rule of thumb (+/-1 agree/disagree encoding, no neutrals): two respondents
        sit exactly ``2 * d / Q`` apart in cosine distance, where ``d`` is the
        number of questions they answer oppositely and ``Q`` is the number of
        questions. So each disagreement moves them ``2/Q`` further apart (0.2 per
        disagreement for Q=10). To split respondents who disagree on ``k`` or more
        questions, set ``threshold ~= (2*k - 1) / Q`` (e.g. Q=10: ~0.3 to split at
        2+ disagreements, ~0.5 at 3+, ~0.7 at 4+). Neutral answers shrink a
        vector rather than flip it, so they dilute (rather than reverse) the
        distance. See ``docs/source/clustering_methods_comparison.md``.
    nan_strategy : {"fill", "ignore"}, optional
        How to handle missing Likert answers. "fill" (default) treats a missing
        answer as neutral, filled with ``fill_value``, so every respondent gets
        clustered. "ignore" excludes any respondent with a missing answer
        (they receive ``-1``, matching this function's behavior prior to the
        ``nan_strategy`` option).
    fill_value : float, optional
        Value used for missing answers when ``nan_strategy="fill"``. Default 0.
    output_column : str, optional
        Name of the cluster-id column to add. Default is "respondent_cluster_id".
    debug : bool, optional
        Forwarded to :func:`encode_likert`. Default False.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with the encoded Likert columns and an integer
        cluster-id column. An all-neutral (zero) response, or (with
        ``nan_strategy="ignore"``) a respondent excluded for a missing answer,
        receives ``-1``. The scipy linkage matrix and the clustered respondent
        index are stored on ``df.attrs["respondent_linkage"]`` /
        ``df.attrs["respondent_linkage_index"]`` so a dendrogram can be drawn (see
        :func:`pandas_survey_toolkit.vis.plot_respondent_dendrogram`).

    Raises
    ------
    ValueError
        If neither 'columns' nor 'pattern' is provided, if both ``n_clusters``
        and ``distance_threshold`` are given, or if ``nan_strategy`` is invalid.

    Notes
    -----
    This method builds an ``n_respondents x n_respondents`` distance matrix, so
    it is meant for small-to-medium surveys. With many thousands of respondents
    prefer :func:`cluster_respondents_umap`, which scales far better;
    :func:`cluster_respondents` chooses between them automatically.
    """
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError("Provide at most one of 'n_clusters' or 'distance_threshold'.")

    columns = _select_likert_columns(df, columns, pattern)

    df = df.encode_likert(columns, custom_mapping=likert_mapping, debug=debug)
    encoded_columns = [f"likert_encoded_{col}" for col in columns]

    df = df.copy()
    df[output_column] = -1

    _check_encoded_range(df, encoded_columns, likert_mapping)
    masked_df = _check_for_nan(
        df, encoded_columns, nan_strategy=nan_strategy, fill_value=fill_value
    )
    responses = masked_df[encoded_columns].astype(float)

    # Cluster the respondents: transpose so each respondent is a column, then
    # cosine distances compare their answer directions across the questions.
    labels, linkage_matrix, clustered, _order = _cosine_cluster(
        responses.T,
        object_label="respondent",
        linkage_method=linkage_method,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
    )

    df.loc[labels.index, output_column] = labels.to_numpy()

    # Expose the linkage so callers can draw a dendrogram. Store it as a plain
    # nested list (not a numpy array): non-scalar objects in ``df.attrs`` break
    # pandas' attrs propagation during melt/concat/groupby.
    df.attrs["respondent_linkage"] = (
        linkage_matrix.tolist() if linkage_matrix is not None else None
    )
    df.attrs["respondent_linkage_index"] = clustered

    return df


@pf.register_dataframe_method
def cluster_survey(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    respondent_method="auto",
    linkage_method="average",
    respondent_n_clusters=None,
    question_n_clusters=None,
    distance_threshold=None,
    size_threshold=1000,
    nan_strategy="fill",
    fill_value=0,
    debug=False,
    **umap_kwargs,
):
    """Cluster a survey on both axes at once (respondents *and* questions).

    Convenience wrapper that encodes the Likert columns, clusters the
    respondents and the questions, and stashes everything the plotting helpers
    need to order both axes. Pass the returned DataFrame straight to
    :func:`pandas_survey_toolkit.vis.cluster_heatmap_plot` (Altair, collapses
    respondents into clusters - good for many respondents / few clusters) or
    :func:`pandas_survey_toolkit.vis.survey_clustermap` (seaborn, per-respondent
    biclustered map - good for few respondents) and both axes come out in a
    sensible, cluster-grouped order.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list, optional
        Question columns. If None, ``pattern`` is used.
    pattern : str, optional
        Regex to match question columns (used if ``columns`` is None).
    likert_mapping : dict, optional
        Passed to :func:`encode_likert`.
    respondent_method : {"auto", "cosine", "umap"}, optional
        How to cluster respondents (same choice as :func:`cluster_respondents`).
        "auto" (default) uses cosine for small surveys (<= ``size_threshold``
        respondents) and UMAP+HDBSCAN above that. An explicit choice that is a
        poor fit for the data size emits a warning (cosine is O(n^2); UMAP needs
        many respondents).
    linkage_method : str, optional
        scipy linkage method for the cosine clustering of both axes. Default
        "average".
    respondent_n_clusters, question_n_clusters : int, optional
        Cut each dendrogram to a fixed number of clusters (per axis).
    distance_threshold : float, optional
        Shared cosine cophenetic-distance cut, applied to whichever axis does not
        have an explicit ``*_n_clusters``.
    size_threshold : int, optional
        Respondent count above which "auto" switches to UMAP. Default 1000.
    nan_strategy : {"fill", "ignore"}, optional
        How to handle missing Likert answers, forwarded to both the respondent
        and question clustering steps. "fill" (default) treats missing answers
        as neutral (see ``fill_value``); "ignore" drops respondents with any
        missing answer instead. See :func:`cluster_respondents_cosine`.
    fill_value : float, optional
        Value used for missing answers when ``nan_strategy="fill"``. Default 0.
    debug : bool, optional
        Forwarded to :func:`encode_likert`. Default False.
    **umap_kwargs
        Extra arguments forwarded to :func:`cluster_respondents_umap` when the
        UMAP path is used.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the encoded Likert columns and an integer
        ``respondent_cluster_id`` column. Question clustering and orderings are
        attached on ``df.attrs`` for the plotting helpers (all plain
        list/dict/scalar objects, so ``out`` stays safe for further pandas ops):

        - ``question_cluster_id`` : dict (question -> cluster id). For the full
          Series with linkage attached, call :func:`cluster_questions` directly.
        - ``question_order`` : encoded question columns in dendrogram order
        - ``question_linkage`` : scipy linkage over the questions (nested list)
        - ``respondent_linkage`` / ``respondent_linkage_index`` : present when
          the cosine respondent method was used
        - ``respondent_method`` : the method actually used

    Raises
    ------
    ValueError
        For an unknown ``respondent_method``, invalid clustering options, or an
        invalid ``nan_strategy``.
    """
    if respondent_method not in ("auto", "cosine", "umap"):
        raise ValueError(
            "respondent_method must be 'auto', 'cosine' or 'umap', got "
            f"{respondent_method!r}."
        )

    columns = _select_likert_columns(df, columns, pattern)
    # With nan_strategy="fill" every respondent ends up clustered, so size
    # selection should count all of them; "ignore" still drops incomplete ones.
    if nan_strategy == "fill":
        n_respondents = int(len(df))
    else:
        n_respondents = int(df[columns].notna().all(axis=1).sum())

    method = respondent_method
    if method == "auto":
        method = _select_clustering_method(n_respondents, size_threshold=size_threshold)
    elif method == "cosine" and n_respondents > size_threshold:
        warnings.warn(
            f"respondent_method='cosine' builds a {n_respondents}x"
            f"{n_respondents} distance matrix, which is expensive/infeasible at "
            "this size; consider 'umap' or 'auto'.",
            stacklevel=2,
        )
    elif method == "umap" and n_respondents < 50:
        warnings.warn(
            f"respondent_method='umap' is unreliable with only {n_respondents} "
            "respondents; consider 'cosine' or 'auto'.",
            stacklevel=2,
        )

    # Distance threshold only applies to axes without an explicit cluster count.
    resp_thr = distance_threshold if respondent_n_clusters is None else None
    ques_thr = distance_threshold if question_n_clusters is None else None

    if method == "cosine":
        out = df.cluster_respondents_cosine(
            columns=columns,
            likert_mapping=likert_mapping,
            linkage_method=linkage_method,
            n_clusters=respondent_n_clusters,
            distance_threshold=resp_thr,
            nan_strategy=nan_strategy,
            fill_value=fill_value,
            output_column="respondent_cluster_id",
            debug=debug,
        )
    else:
        out = df.cluster_respondents_umap(
            columns=columns,
            likert_mapping=likert_mapping,
            nan_strategy=nan_strategy,
            fill_value=fill_value,
            output_columns=(
                "respondent_cluster_id",
                "respondent_cluster_probability",
            ),
            debug=debug,
            **umap_kwargs,
        )

    # Cluster the questions (cosine, on the same encoded columns).
    question_clusters = out.cluster_questions(
        columns=columns,
        likert_mapping=likert_mapping,
        linkage_method=linkage_method,
        n_clusters=question_n_clusters,
        distance_threshold=ques_thr,
        nan_strategy=nan_strategy,
        fill_value=fill_value,
        debug=False,
    )

    # Store only plain (list/dict/scalar) objects in attrs so downstream pandas
    # operations on ``out`` (melt/concat/groupby) keep working. The full Series
    # is available directly from ``cluster_questions`` if needed.
    q_linkage = question_clusters.attrs.get("linkage")
    out.attrs["question_cluster_id"] = {
        str(q): int(c) for q, c in question_clusters.items()
    }
    out.attrs["question_order"] = list(question_clusters.attrs.get("encoded_order", []))
    out.attrs["question_linkage"] = (
        q_linkage.tolist() if q_linkage is not None else None
    )
    out.attrs["respondent_method"] = method
    return out


@pf.register_dataframe_method
def encode_likert(
    df,
    likert_columns,
    output_prefix="likert_encoded_",
    custom_mapping=None,
    debug=True,
):
    """Encode Likert scale responses to numeric values.

    Responses are encoded on a 3-point scale (-1 / 0 / +1). This intentionally
    ignores intensity (both "agree" and "strongly agree" map to +1): the
    clustering uses cosine distance on these vectors, which cares about the
    *direction* of a respondent's opinions, not their magnitude.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    likert_columns : list
        List of column names containing Likert scale responses.
    output_prefix : str, optional
        Prefix for the new encoded columns. Default is ``likert_encoded_``.
    custom_mapping : dict, optional
        Optional custom mapping for Likert scale responses. If provided, the
        built-in mapping is ignored.
    debug : bool, optional
        If True, prints out the mappings. Default is True.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for encoded Likert responses.

    Notes
    -----
    Default mapping:
    - -1: Phrases containing 'disagree', 'do not agree', etc.
    - 0: Phrases containing 'neutral', 'neither', 'unsure', etc.
    - +1: Phrases containing 'agree' (but not 'disagree' or 'not agree')
    - NaN: NaN values are preserved
    """
    df = df.copy()

    # Values a valid built-in mapping is allowed to produce.
    valid_values = {-1, 0, 1}

    def default_mapping(response):
        if pd.isna(response):
            return pd.NA
        response = str(response).lower().strip()

        # Neutral / Neither / Unsure / Don't know (0)
        if re.search(r"\b(neutral|neither|unsure|know)\b", response) or re.search(
            r"neither\s+agree\s+nor\s+disagree", response
        ):
            return 0

        # Disagree / Dissatisfied (-1)
        if re.search(r"\b(disagree)\b", response) or re.search(
            r"\b(dis|not|no)[-]{0,1}\s*(agree|satisf)", response
        ):
            return -1

        # Agree / Satisfied (1)
        if re.search(r"\bagree\b", response) or re.search(r"satisf", response):
            return 1

        # Unable to classify
        return None

    conversion_summary = defaultdict(int)
    unconverted_phrases = set()

    if custom_mapping is None:
        mapping_func = default_mapping
        if debug:
            print("Using default mapping:")
            print("-1: Phrases containing 'disagree', 'do not agree', etc.")
            print(" 0: Phrases containing 'neutral', 'neither', 'unsure', etc.")
            print("+1: Phrases containing 'agree' (but not 'disagree' or 'not agree')")
            print("NaN: NaN values are preserved")
    else:

        def mapping_func(response):
            if pd.isna(response):
                return pd.NA
            converted = custom_mapping.get(str(response).lower().strip())
            if converted is None:
                unconverted_phrases.add(str(response))
                return pd.NA
            return converted

        if debug:
            print("Using custom mapping:", custom_mapping)
            print("NaN: NaN values are preserved")

    for column in likert_columns:
        output_column = f"{output_prefix}{column}"
        df[output_column] = df[column].apply(lambda x: mapping_func(x))

        # Update conversion summary
        for original, converted in zip(df[column], df[output_column]):
            conversion_summary[f"{original} -> {converted}"] += 1

    if debug:
        for conversion, count in conversion_summary.items():
            print(f"  {conversion}: {count} times")

    # Alert about unconverted phrases
    if unconverted_phrases:
        warnings.warn(
            "The following phrases were not converted (mapped to NaN): "
            f"{', '.join(unconverted_phrases)}",
            stacklevel=2,
        )

    # Alert if default mapping didn't convert everything
    if custom_mapping is None:
        all_responses = set()
        for column in likert_columns:
            all_responses.update(df[column].dropna().unique())
        unconverted = [
            resp for resp in all_responses if default_mapping(resp) not in valid_values
        ]
        if unconverted:
            warnings.warn(
                "The default mapping didn't convert the following responses: "
                f"{', '.join(unconverted)}",
                stacklevel=2,
            )

    return df


@pf.register_dataframe_method
def extract_keywords(
    df: pd.DataFrame,
    input_column: str,
    output_column: str = "keywords",
    preprocessed_column: str = "preprocessed_text",
    spacy_column: str = "spacy_output",
    lemma_column: str = "lemmatized_text",
    top_n: int = 3,
    threshold: float = 0.4,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 5,
    min_count: int = None,
    min_proportion_with_keywords: float = 0.95,
    **kwargs,
) -> pd.DataFrame:
    """Apply a pipeline of text preprocessing, spaCy processing, lemmatization,
    and TF-IDF to extract keywords from the specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to process.
    output_column : str, optional
        Name of the column to store the extracted keywords. Default is 'keywords'.
    preprocessed_column : str, optional
        Name of the column to store preprocessed text. Default is 'preprocessed_text'.
    spacy_column : str, optional
        Name of the column to store spaCy output. Default is 'spacy_output'.
    lemma_column : str, optional
        Name of the column to store lemmatized text. Default is 'lemmatized_text'.
    top_n : int, optional
        Number of top keywords to extract for each document. Default is 3.
    threshold : float, optional
        Minimum TF-IDF score for a keyword to be included. Default is 0.4.
    ngram_range : tuple, optional
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. Default is (1, 1) which means only unigrams.
    min_df : int, optional
        Minimum document frequency for TF-IDF. Default is 5.
    min_count : int, optional
        Minimum count for a keyword to be considered common in refinement.
        Default is None.
    min_proportion_with_keywords : float, optional
        Minimum proportion of rows that should have keywords after refinement.
        Default is 0.95.
    **kwargs
        Additional keyword arguments to pass to the preprocessing, spaCy,
        lemmatization, or TF-IDF functions.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for preprocessed text,
        spaCy output, lemmatized text, and extracted keywords.
    """
    df_temp = df.copy()
    # Step 1: Preprocess text
    df_temp = df_temp.preprocess_text(
        input_column=input_column,
        output_column=preprocessed_column,
        **kwargs.get("preprocess_kwargs", {}),
    )

    df_temp = df_temp.remove_short_comments(
        input_column=input_column, min_comment_length=5
    )

    # Step 2: Apply spaCy
    df_temp = df_temp.fit_spacy(
        input_column=preprocessed_column, output_column=spacy_column
    )

    # Step 3: Get lemmatized text
    df_temp = df_temp.get_lemma(
        input_column=spacy_column,
        output_column=lemma_column,
        **kwargs.get("lemma_kwargs", {}),
    )

    # Step 4: Apply TF-IDF and extract keywords
    df_temp = df_temp.fit_tfidf(
        input_column=lemma_column,
        output_column=output_column,
        top_n=top_n,
        threshold=threshold,
        ngram_range=ngram_range,
        min_df=min_df,
        **kwargs.get("tfidf_kwargs", {}),
    )

    df_temp = df_temp.refine_keywords(
        keyword_column=output_column,
        text_column=lemma_column,
        min_proportion=min_proportion_with_keywords,
        output_column="refined_keywords",
        min_count=min_count,
    )

    return df_temp


@pf.register_dataframe_method
def refine_keywords(
    df: pd.DataFrame,
    keyword_column: str = "keywords",
    text_column: str = "lemmatized_text",
    min_count: Union[int, None] = None,
    min_proportion: float = 0.95,
    output_column: str = None,
    debug: bool = True,
) -> pd.DataFrame:
    """Refine keywords by replacing rare keywords with more common ones based
    on the text content.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    keyword_column : str, optional
        Name of the column containing keyword lists. Default is 'keywords'.
    text_column : str, optional
        Name of the column containing the original text. Default is 'lemmatized_text'.
    min_count : int, optional
        Minimum count for a keyword to be considered common. If None,
        it will be determined automatically. Default is None.
    min_proportion : float, optional
        Minimum proportion of rows that should have keywords after refinement.
        Used only if min_count is None. Default is 0.95.
    output_column : str, optional
        Column name for the refined keyword output. If None, the keyword_column
        is overwritten. Default is None.
    debug : bool, optional
        If True, print detailed statistics about the refinement process.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with refined keywords.
    """
    if output_column is None:
        output_column = keyword_column

    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [keyword_column, text_column])

    # Step 1 & 2: Collect all keywords and count them
    all_keywords = [
        keyword
        for keywords in masked_df[keyword_column]
        if isinstance(keywords, list)
        for keyword in keywords
    ]
    keyword_counts = pd.Series(all_keywords).value_counts()

    def refine_row_keywords(row, common_keywords):
        if pd.isna(row[text_column]) or not isinstance(row[keyword_column], list):
            return []

        text = str(row[text_column]).lower()
        current_keywords = row[keyword_column]
        refined_keywords = []

        for keyword in current_keywords:
            if keyword in common_keywords:
                refined_keywords.append(keyword)
            else:
                # Find a replacement from common keywords
                for common_keyword in sorted(
                    common_keywords, key=lambda k: (-keyword_counts[k], len(k))
                ):
                    if (
                        common_keyword in text
                        and common_keyword not in refined_keywords
                    ):
                        refined_keywords.append(common_keyword)
                        break

        # Ensure correct ordering based on appearance in the original text
        return (
            sorted(refined_keywords, key=lambda k: text.index(k))
            if refined_keywords
            else []
        )

    if min_count is None:
        # Determine min_count automatically
        def get_proportion_with_keywords(count):
            common_keywords = set(keyword_counts[keyword_counts >= count].index)
            refined_keywords = masked_df.apply(
                lambda row: refine_row_keywords(row, common_keywords), axis=1
            )
            return (refined_keywords.str.len() > 0).mean()

        min_count = 1
        while get_proportion_with_keywords(min_count) > min_proportion:
            min_count += 1
        min_count -= 1  # Go back one step to ensure we're above the min_proportion

    # Separate common and rare keywords
    common_keywords = set(keyword_counts[keyword_counts >= min_count].index)

    # Apply the refinement to each row
    masked_df[output_column] = masked_df.apply(
        lambda row: refine_row_keywords(row, common_keywords), axis=1
    )

    # Combine results
    df_to_return = combine_results(df, masked_df, mask, [output_column])

    if debug:
        # Calculate statistics
        original_keyword_count = masked_df[keyword_column].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        refined_keyword_count = masked_df[output_column].apply(len)

        original_unique_keywords = set(
            keyword
            for keywords in masked_df[keyword_column]
            if isinstance(keywords, list)
            for keyword in keywords
        )
        refined_unique_keywords = set(
            keyword for keywords in masked_df[output_column] for keyword in keywords
        )

        print(f"Refinement complete. Min count used: {min_count}")
        print(f"Original average keywords per row: {original_keyword_count.mean():.2f}")
        print(f"Refined average keywords per row: {refined_keyword_count.mean():.2f}")
        print(
            "Proportion of rows with keywords after refinement: "
            f"{(refined_keyword_count > 0).mean():.2%}"
        )
        print(
            f"Total unique keywords before refinement: {len(original_unique_keywords)}"
        )
        print(f"Total unique keywords after refinement: {len(refined_unique_keywords)}")
        print(
            "Reduction in unique keywords: "
            f"{(1 - len(refined_unique_keywords) / len(original_unique_keywords)):.2%}"
        )

    return df_to_return


@pf.register_dataframe_method
def clean_survey_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame column names for use with MS Forms / Excel exports.

    Column names from MS Forms are full question texts and often contain
    characters that break regex patterns (?, (), /) or are otherwise awkward
    to work with. This function lowercases, removes punctuation, normalises
    whitespace, and replaces spaces with underscores.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with cleaned column names.

    Examples
    --------
    >>> df.clean_survey_columns()
    # "How satisfied are you? (1-5)" -> "how_satisfied_are_you_1_5"
    # "Don't Know / Pass"            -> "dont_know_pass"
    """
    col_df = pd.DataFrame({"col": df.columns.tolist()})
    col_df = col_df.preprocess_text(
        input_column="col",
        output_column="clean_col",
        lower_case=True,
        remove_punctuation=True,
        keep_sentence_punctuation=False,
        normalize_whitespace=True,
    )
    new_names = (
        col_df["clean_col"]
        .str.replace("'", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)  # re-normalise after punctuation gaps
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .tolist()
    )

    # Deduplicate: append _1, _2, ... when cleaning produces identical names
    seen: dict = {}
    deduped = []
    for name in new_names:
        if name not in seen:
            seen[name] = 0
            deduped.append(name)
        else:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")

    return df.rename(columns=dict(zip(df.columns, deduped)))


@pf.register_dataframe_method
def remove_short_comments(
    df: pd.DataFrame, input_column: str, min_comment_length: int = 5
) -> pd.DataFrame:
    """Replace comments shorter than the specified minimum length with NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to process.
    min_comment_length : int, optional
        Minimum length of comment to keep. Default is 5.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with short comments replaced by NaN.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Replace short comments with NaN
    df_copy[input_column] = df_copy[input_column].apply(
        lambda x: x if isinstance(x, str) and len(x) >= min_comment_length else np.nan
    )

    return df_copy


@pf.register_dataframe_method
def fit_sentence_transformer(
    df,
    input_column: str,
    model_name="all-MiniLM-L6-v2",
    output_column="sentence_embedding",
):
    """Add vector embeddings for each string in the input column.

    Creates sentence embeddings that can be used for downstream tasks like clustering.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to embed.
    model_name : str, optional
        Name of the sentence transformer model to use. Default is 'all-MiniLM-L6-v2'.
    output_column : str, optional
        Name of the column to store embeddings. Default is 'sentence_embedding'.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing sentence embeddings.
    """

    # Initialize the sentence transformer model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _require_nlp_extra("fit_sentence_transformer")
    masked_df, mask = create_masked_df(df, [input_column])
    model = SentenceTransformer(model_name)

    # Create sentence embeddings
    embeddings = model.encode(masked_df[input_column].tolist())

    # Convert embeddings to a list of numpy arrays
    embeddings_list = [embedding for embedding in embeddings]

    # Add the embeddings as a new column in the dataframe
    masked_df[output_column] = embeddings_list
    df_to_return = combine_results(df, masked_df, mask, output_column)

    return df_to_return


@pf.register_dataframe_method
def extract_sentiment(
    df,
    input_column: str,
    output_columns=["positive", "neutral", "negative", "sentiment"],
):
    """Extract sentiment from text using the
    cardiffnlp/twitter-roberta-base-sentiment model.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to analyze.
    output_columns : list, optional
        List of column names for the output.
        Default is ["positive", "neutral", "negative", "sentiment"].

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for sentiment scores and labels.
    """

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError:
        _require_nlp_extra("extract_sentiment")

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    masked_df, mask = create_masked_df(df, [input_column])

    def analyze_sentiment(text):
        encoded_input = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        return scores

    sentiment_scores = masked_df[input_column].apply(analyze_sentiment)

    masked_df[output_columns[0]] = sentiment_scores.apply(lambda x: x[2])  # Positive
    masked_df[output_columns[1]] = sentiment_scores.apply(lambda x: x[1])  # Neutral
    masked_df[output_columns[2]] = sentiment_scores.apply(lambda x: x[0])  # Negative

    masked_df[output_columns[3]] = masked_df[
        [output_columns[0], output_columns[1], output_columns[2]]
    ].idxmax(axis=1)
    masked_df[output_columns[3]] = masked_df[output_columns[3]].map(
        {
            output_columns[0]: "positive",
            output_columns[1]: "neutral",
            output_columns[2]: "negative",
        }
    )

    df_to_return = combine_results(df, masked_df, mask, output_columns)
    return df_to_return


@pf.register_dataframe_method
def cluster_comments(
    df: pd.DataFrame,
    input_column: str,
    output_columns: str = ["cluster", "cluster_probability"],
    min_cluster_size=5,
    cluster_selection_epsilon: float = 0.2,
    n_neighbors: int = 15,
):
    """Apply a pipeline for clustering text comments.

    Applies a pipeline of:
    1) Vector embeddings
    2) Dimensional reduction
    3) Clustering

    This assigns each row a cluster ID so that similar free text comments
    (found in the input_column) can be grouped together.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to cluster.
    output_columns : list, optional
        Names for the output columns. Default is ["cluster", "cluster_probability"].
    min_cluster_size : int, optional
        The minimum size of clusters for HDBSCAN. Default is 5.
    cluster_selection_epsilon : float, optional
        Distance threshold for HDBSCAN. Higher epsilon means fewer, larger clusters.
        Default is 0.2.
    n_neighbors : int, optional
        The size of local neighborhood for UMAP. Default is 15.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for cluster IDs and probabilities.
    """

    df_temp = (
        df.fit_sentence_transformer(
            input_column=input_column, output_column="sentence_embedding"
        )
        .fit_umap(
            input_columns="sentence_embedding",
            embeddings_in_list=True,
            n_neighbors=n_neighbors,
        )
        .fit_cluster_hdbscan(
            output_columns=output_columns,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
    )

    return df_temp


@pf.register_dataframe_method
def fit_tfidf(
    df: pd.DataFrame,
    input_column: str,
    output_column: str = "keywords",
    top_n: int = 3,
    threshold: float = 0.6,
    append_features: bool = False,
    ngram_range: Tuple[int, int] = (1, 1),
    **tfidf_kwargs,
) -> pd.DataFrame:
    """Apply TF-IDF vectorization to extract top keywords from text.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to vectorize.
    output_column : str, optional
        Name of the column to store the extracted keywords. Default is 'keywords'.
    top_n : int, optional
        Number of top keywords to extract for each document. Default is 3.
    threshold : float, optional
        Minimum TF-IDF score for a keyword to be included. Default is 0.6.
    append_features : bool, optional
        If True, append all TF-IDF features to the DataFrame (useful for
        downstream machine learning tasks). Default is False.
    ngram_range : tuple, optional
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. Default is (1, 1) which means only unigrams.
        Set to (1, 2) for unigrams and bigrams, and so on.
    **tfidf_kwargs
        Additional keyword arguments to pass to TfidfVectorizer.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing the top keywords.
    """
    # Create a masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])

    if masked_df.empty:
        result = df.copy()
        result[output_column] = np.nan
        return result

    # Ensure ngram_range is included in the TfidfVectorizer parameters
    tfidf_kwargs["ngram_range"] = ngram_range
    # Inside fit_tfidf function
    tfidf_kwargs["min_df"] = tfidf_kwargs.get("min_df", 1)

    # Apply TF-IDF vectorization to the masked DataFrame
    tfidf_features, _, feature_names = apply_vectorizer(
        masked_df, input_column, vectorizer_name="TfidfVectorizer", **tfidf_kwargs
    )

    def extract_top_keywords(row: pd.Series) -> List[str]:
        # Get indices of top N TF-IDF scores
        top_indices = row.nlargest(top_n).index

        # Get the original text for this row
        original_text = masked_df.loc[row.name, input_column].lower()

        # Filter based on threshold and presence in original text
        top_keywords = [
            feature_names[i]
            for i, idx in enumerate(tfidf_features.columns)
            if idx in top_indices
            and row[idx] >= threshold
            and feature_names[i].lower() in original_text
        ]

        # Sort keywords based on their order in the original text
        return sorted(top_keywords, key=lambda x: original_text.index(x.lower()))

    # Extract top keywords for each document
    masked_df[output_column] = tfidf_features.apply(extract_top_keywords, axis=1)

    # Combine the results back into the original DataFrame
    result_df = combine_results(df, masked_df, mask, [output_column])

    # Optionally append all TF-IDF features
    if append_features:
        # We need to handle NaN values in the features as well
        feature_columns = tfidf_features.columns.tolist()
        masked_df = pd.concat([masked_df, tfidf_features], axis=1)
        result_df = combine_results(result_df, masked_df, mask, feature_columns)

    return result_df


@pf.register_dataframe_method
def fit_spacy(df, input_column: str, output_column: str = "spacy_output"):
    """Apply the en_core_web_md spaCy model to the specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to analyze.
    output_column : str, optional
        Name of the output column. Default is "spacy_output".

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing spaCy doc objects.

    Notes
    -----
    If the spaCy model is not already downloaded, this function will attempt
    to download it automatically.
    """

    try:
        import spacy
    except ImportError:
        _require_nlp_extra("fit_spacy")

    # Check if the model is downloaded, if not, download it
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Downloading en_core_web_md model...")
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")

    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])

    # Apply spaCy model
    masked_df[output_column] = masked_df[input_column].apply(nlp)

    # Combine results
    df_to_return = combine_results(df, masked_df, mask, output_column)

    return df_to_return


@pf.register_dataframe_method
def get_lemma(
    df: pd.DataFrame,
    input_column: str = "spacy_output",
    output_column: str = "lemmatized_text",
    text_pos: List[str] = ["PRON"],
    remove_punct: bool = True,
    remove_space: bool = True,
    remove_stop: bool = True,
    keep_tokens: Union[List[str], None] = None,
    keep_pos: Union[List[str], None] = None,
    keep_dep: Union[List[str], None] = ["neg"],
    join_tokens: bool = True,
) -> pd.DataFrame:
    """Extract lemmatized text from spaCy doc objects.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str, optional
        Name of the column containing spaCy doc objects. Default is 'spacy_output'.
    output_column : str, optional
        Name of the output column for lemmatized text. Default is 'lemmatized_text'.
    text_pos : List[str], optional
        List of POS tags to exclude from lemmatization and return the text.
        Default is ['PRON'].
    remove_punct : bool, optional
        Whether to remove punctuation. Default is True.
    remove_space : bool, optional
        Whether to remove whitespace tokens. Default is True.
    remove_stop : bool, optional
        Whether to remove stop words. Default is True.
    keep_tokens : List[str], optional
        List of token texts to always keep. Default is None.
    keep_pos : List[str], optional
        List of POS tags to always keep. Default is None.
    keep_dep : List[str], optional
        List of dependency labels to always keep. Default is ["neg"].
    join_tokens : bool, optional
        Whether to join tokens into a string. If False, returns a list of tokens.
        Default is True.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing lemmatized
        text or token list.
    """

    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])

    def remove_token(token):
        """
        Returns True if the token should be removed.
        """
        if (
            (keep_tokens and token.text in keep_tokens)
            or (keep_pos and token.pos_ in keep_pos)
            or (keep_dep and token.dep_ in keep_dep)
        ):
            return False
        return (
            (remove_punct and token.is_punct)
            or (remove_space and token.is_space)
            or (remove_stop and token.is_stop)
        )

    def process_text(doc):
        tokens = [
            token.text if token.pos_ in text_pos else token.lemma_
            for token in doc
            if not remove_token(token)
        ]
        return " ".join(tokens) if join_tokens else tokens

    # Apply processing
    masked_df[output_column] = masked_df[input_column].apply(process_text)

    # Combine results
    df_to_return = combine_results(df, masked_df, mask, output_column)

    return df_to_return


@pf.register_dataframe_method
def preprocess_text(
    df: pd.DataFrame,
    input_column: str,
    output_column: str = None,
    remove_html: bool = True,
    lower_case: bool = False,
    normalize_whitespace: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    flag_short_comments: bool = False,
    min_comment_length: int = 5,
    max_comment_length: int = None,
    remove_punctuation: bool = True,
    keep_sentence_punctuation: bool = True,
    comment_length_column: str = None,
) -> pd.DataFrame:
    """Preprocess text data in the specified column, tailored for survey responses.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    input_column : str
        Name of the column containing text to preprocess.
    output_column : str, optional
        Name of the output column. If None, overwrites the input column.
    remove_html : bool, optional
        Whether to remove unexpected HTML tags. Default is True.
    lower_case : bool, optional
        Whether to lowercase all words. Default is False.
    normalize_whitespace : bool, optional
        Whether to normalize whitespace. Default is True.
    remove_numbers : bool, optional
        Whether to remove numbers. Default is False.
    remove_stopwords : bool, optional
        Whether to remove stop words. Default is False.
    flag_short_comments : bool, optional
        Whether to flag very short comments. Default is False.
    min_comment_length : int, optional
        Minimum length of comment to not be flagged as short. Default is 5.
    max_comment_length : int, optional
        Maximum length of comment to keep. If None, keeps full length. Default is None.
    remove_punctuation : bool, optional
        Whether to remove punctuation. Default is True.
    keep_sentence_punctuation : bool, optional
        Whether to keep sentence-level punctuation. Default is True.
    comment_length_column : str, optional
        Name of the column to store comment lengths. If None, no column is
        added. Default is None.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with preprocessed text and optionally new columns for
        short comments, truncation info, and comment length.
    """

    output_column = output_column or input_column

    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])

    def process_text(text):
        if lower_case:
            text = text.lower()
        if remove_html:
            text = strip_tags(text)

        if normalize_whitespace:
            text = strip_multiple_whitespaces(text)

        if remove_numbers:
            text = strip_numeric(text)

        if remove_stopwords:
            text = gensim_remove_stopwords(text)

        if remove_punctuation:
            if keep_sentence_punctuation:
                # Remove all punctuation except .,!?'" and apostrophes
                text = re.sub(r"[^\w\s.,!?'\"]", "", text)
                # Remove spaces before punctuation, but not before apostrophes
                text = re.sub(r"\s([.,!?\"](?:\s|$))", r"\1", text)
            else:
                # Remove all punctuation except apostrophes
                text = re.sub(r"[^\w\s']", "", text)

        text = text.strip()

        if max_comment_length:
            text = text[:max_comment_length]

        return text

    # Apply processing
    masked_df[output_column] = masked_df[input_column].apply(process_text)

    columns_to_combine = [output_column]

    if flag_short_comments:
        short_comment_col = f"{output_column}_is_short"
        masked_df[short_comment_col] = (
            masked_df[output_column].str.len() < min_comment_length
        )
        columns_to_combine.append(short_comment_col)

    if max_comment_length:
        truncated_col = f"{output_column}_was_truncated"
        masked_df[truncated_col] = (
            masked_df[input_column].str.len() > max_comment_length
        )
        columns_to_combine.append(truncated_col)

    if comment_length_column:
        masked_df[comment_length_column] = masked_df[output_column].str.len()
        columns_to_combine.append(comment_length_column)

    # Combine results
    df_to_return = combine_results(df, masked_df, mask, columns_to_combine)

    return df_to_return
