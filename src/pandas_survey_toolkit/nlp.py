import re
import warnings
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
import spacy
from gensim.parsing.preprocessing import (
    remove_stopwords as gensim_remove_stopwords,
)
from gensim.parsing.preprocessing import (
    strip_multiple_whitespaces,
    strip_numeric,
    strip_tags,
)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Importing analytics registers the fit_umap / fit_cluster_hdbscan dataframe
# methods that cluster_respondents relies on.
from pandas_survey_toolkit import analytics  # noqa: F401
from pandas_survey_toolkit.utils import (
    apply_vectorizer,
    combine_results,
    create_masked_df,
)


def _select_likert_columns(df, columns, pattern):
    """Resolve the list of Likert columns from an explicit list or a regex."""
    if columns is None and pattern is None:
        raise ValueError("Either 'columns' or 'pattern' must be provided.")
    if columns is None:
        columns = df.filter(regex=pattern).columns.tolist()
    return columns


@pf.register_dataframe_method
def cluster_respondents(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    scale=3,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    hdbscan_min_cluster_size=20,
    hdbscan_min_samples=None,
    cluster_selection_epsilon=0.4,
    output_columns=("respondent_cluster_id", "respondent_cluster_probability"),
    debug=False,
):
    """Cluster *respondents* by their Likert response patterns (UMAP + HDBSCAN).

    Each respondent (row) is turned into a vector of encoded answers and embedded
    with UMAP (cosine metric), then grouped with HDBSCAN. Because UMAP embeds one
    point *per respondent*, this method needs a reasonable number of respondents
    to work well. When respondents are few (roughly, fewer than the number of
    questions) prefer :func:`cluster_respondents_correlation`. See
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
        Custom mapping for Likert scale responses. If None, the built-in
        ``scale``-point mapping is used.
    scale : int, optional
        Encoding scale passed to :func:`encode_likert` (3 or 5). Default is 3.
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
    output_columns : tuple, optional
        Names for the (cluster id, cluster probability) output columns. Default
        is ("respondent_cluster_id", "respondent_cluster_probability").

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for encoded Likert responses,
        UMAP coordinates, and cluster IDs.

    Raises
    ------
    ValueError
        If neither 'columns' nor 'pattern' is provided.
    """
    columns = _select_likert_columns(df, columns, pattern)

    # Encode Likert scales
    df = df.encode_likert(
        columns, custom_mapping=likert_mapping, scale=scale, debug=debug
    )
    encoded_columns = [f"likert_encoded_{col}" for col in columns]

    # Guard against surveys that are too small for density-based clustering.
    # HDBSCAN raises if min_samples exceeds the number of points, and cannot
    # form a cluster of ``min_cluster_size`` if there are fewer respondents than
    # that. Shrink the parameters (with a warning) so the call degrades to a
    # trivial result instead of erroring, and point users at the correlation
    # method which is designed for this regime.
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
            "respondents - consider cluster_respondents_correlation, which is "
            "designed for surveys with few respondents and many questions."
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
def cluster_questions(df, **kwargs):
    """Deprecated alias for :func:`cluster_respondents`.

    Despite its name this always clustered *respondents* (one UMAP point per
    row), not questions. It is retained for backwards compatibility and emits a
    ``DeprecationWarning``; the output columns keep their historical names
    (``question_cluster_id`` / ``question_cluster_probability``). New code should
    call :func:`cluster_respondents` or :func:`cluster_respondents_correlation`.
    """
    warnings.warn(
        "cluster_questions is deprecated and clusters respondents, not questions."
        " Use cluster_respondents (UMAP+HDBSCAN) or"
        " cluster_respondents_correlation (better for few respondents) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return df.cluster_respondents(
        output_columns=("question_cluster_id", "question_cluster_probability"),
        **kwargs,
    )


@pf.register_dataframe_method
def cluster_respondents_correlation(
    df,
    columns=None,
    pattern=None,
    likert_mapping=None,
    scale=3,
    corr_method="pearson",
    linkage_method="average",
    distance="signed",
    n_clusters=None,
    distance_threshold=None,
    output_column="respondent_cluster_id",
    debug=False,
):
    """Cluster respondents using correlation + hierarchical (dendrogram) clustering.

    This is the recommended approach when there are **few respondents relative to
    the number of questions**. Rather than embedding respondents as points (which
    UMAP/HDBSCAN need many of), it correlates every pair of respondents *across
    the questions* and runs agglomerative clustering on the resulting distance
    matrix. Each correlation is estimated over all the questions, so the estimates
    stay stable even when respondents are scarce. See
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
        Custom mapping for Likert scale responses. If None, the built-in
        ``scale``-point mapping is used.
    scale : int, optional
        Encoding scale passed to :func:`encode_likert` (3 or 5). Default is 3.
        Use 5 if the intensity of agreement should influence the grouping.
    corr_method : str, optional
        Correlation method passed to :meth:`pandas.DataFrame.corr`
        ("pearson", "spearman" or "kendall"). Default is "pearson", which
        centres each respondent and therefore groups by response *shape* rather
        than overall positivity (acquiescence bias).
    linkage_method : str, optional
        Linkage method for :func:`scipy.cluster.hierarchy.linkage`
        ("average", "complete", "single", ...). Default is "average".
        (Ward is not offered because it requires raw Euclidean coordinates, not a
        precomputed correlation-distance matrix.)
    distance : str, optional
        How to turn a correlation ``r`` into a distance. "signed" -> ``1 - r``
        (opposite responders are far apart; the default), or "absolute" ->
        ``1 - |r|`` (group by strength of association regardless of sign).
    n_clusters : int, optional
        If given, cut the dendrogram to produce exactly this many clusters
        (``criterion="maxclust"``). Mutually exclusive with ``distance_threshold``.
    distance_threshold : float, optional
        If given, cut the dendrogram at this cophenetic distance
        (``criterion="distance"``). If neither ``n_clusters`` nor
        ``distance_threshold`` is provided, a threshold of 1.0 is used (which
        splits positively- from non-positively-correlated respondents).
    output_column : str, optional
        Name of the cluster-id column to add. Default is "respondent_cluster_id".

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with the encoded Likert columns and an integer
        cluster-id column. Respondents that could not be clustered (NaN answers,
        or constant answers giving undefined correlation) receive ``-1``. The
        scipy linkage matrix and the clustered respondent index are stored on
        ``df.attrs["respondent_linkage"]`` / ``df.attrs["respondent_linkage_index"]``
        so a dendrogram can be drawn (see
        :func:`pandas_survey_toolkit.vis.plot_respondent_dendrogram`).

    Raises
    ------
    ValueError
        If neither 'columns' nor 'pattern' is provided, if both ``n_clusters``
        and ``distance_threshold`` are given, or for an unknown ``distance``.
    """
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError("Provide at most one of 'n_clusters' or 'distance_threshold'.")
    if distance not in ("signed", "absolute"):
        raise ValueError(f"distance must be 'signed' or 'absolute', got {distance!r}.")

    columns = _select_likert_columns(df, columns, pattern)

    df = df.encode_likert(
        columns, custom_mapping=likert_mapping, scale=scale, debug=debug
    )
    encoded_columns = [f"likert_encoded_{col}" for col in columns]

    df = df.copy()
    df[output_column] = -1

    # Keep only respondents with a complete set of answers.
    masked_df, mask = create_masked_df(df, encoded_columns)
    responses = masked_df[encoded_columns].astype(float)

    # Respondents with no variation across questions have an undefined
    # correlation with everyone; exclude them (they stay labelled -1).
    varying = responses.std(axis=1) > 0
    if not varying.all():
        warnings.warn(
            f"{int((~varying).sum())} respondent(s) gave the same answer to every "
            "question; their correlation is undefined so they are left unclustered "
            "(cluster -1)."
        )
    responses = responses[varying]

    if len(responses) < 2:
        warnings.warn(
            "Fewer than two respondents can be correlated; no clusters formed."
        )
        return df

    # Correlate respondents against each other across the questions.
    corr = responses.T.corr(method=corr_method)

    if distance == "signed":
        dist = 1.0 - corr
    else:  # "absolute"
        dist = 1.0 - corr.abs()

    # Guard against tiny floating-point artefacts before condensing.
    dist_values = dist.to_numpy()
    dist_values = (dist_values + dist_values.T) / 2.0
    np.fill_diagonal(dist_values, 0.0)
    dist_values = np.clip(dist_values, 0.0, None)

    condensed = squareform(dist_values, checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)

    if n_clusters is not None:
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    else:
        threshold = 1.0 if distance_threshold is None else distance_threshold
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    df.loc[responses.index, output_column] = labels.astype(int)

    # Expose the linkage so callers can draw a dendrogram.
    df.attrs["respondent_linkage"] = linkage_matrix
    df.attrs["respondent_linkage_index"] = list(responses.index)

    return df


@pf.register_dataframe_method
def encode_likert(
    df,
    likert_columns,
    output_prefix="likert_encoded_",
    custom_mapping=None,
    scale=3,
    debug=True,
):
    """Encode Likert scale responses to numeric values.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    likert_columns : list
        List of column names containing Likert scale responses.
    output_prefix : str, optional
        Prefix for the new encoded columns. Default is 'likert_encoded_'.
    custom_mapping : dict, optional
        Optional custom mapping for Likert scale responses. If provided, the
        built-in ``scale`` mapping is ignored.
    scale : int, optional
        Number of points for the built-in mapping. Either 3 (default) or 5.

        - ``3``: agree/disagree collapse to +1/-1 regardless of intensity.
        - ``5``: intensity is preserved, so "strongly agree" -> +2,
          "agree" -> +1, neutral -> 0, "disagree" -> -1,
          "strongly disagree" -> -2. Use this when the *magnitude* of
          sentiment should count towards the distance between respondents
          (see the encoding discussion in
          ``docs/source/clustering_methods_comparison.md``).
    debug : bool, optional
        If True, prints out the mappings. Default is True.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for encoded Likert responses.

    Notes
    -----
    Default (3-point) mapping:
    - -1: Phrases containing 'disagree', 'do not agree', etc.
    - 0: Phrases containing 'neutral', 'neither', 'unsure', etc.
    - +1: Phrases containing 'agree' (but not 'disagree' or 'not agree')
    - NaN: NaN values are preserved

    With ``scale=5`` an "intensifier" (strongly/very/completely/totally/
    extremely/highly) on an agree/disagree phrase pushes the value out to
    +2 / -2.
    """
    if scale not in (3, 5):
        raise ValueError(f"scale must be 3 or 5, got {scale}.")

    df = df.copy()

    # Values a valid built-in mapping is allowed to produce.
    valid_values = {-1, 0, 1} if scale == 3 else {-2, -1, 0, 1, 2}

    def default_mapping(response):
        if pd.isna(response):
            return pd.NA
        response = str(response).lower().strip()

        # Neutral / Neither / Unsure / Don't know (0)
        if re.search(r"\b(neutral|neither|unsure|know)\b", response) or re.search(
            r"neither\s+agree\s+nor\s+disagree", response
        ):
            return 0

        # Intensity modifier (only affects the 5-point scale)
        strong = bool(
            re.search(
                r"\b(strongly|very|completely|totally|extremely|highly)\b", response
            )
        )

        # Disagree / Dissatisfied (-1, or -2 when strong on the 5-point scale)
        if re.search(r"\b(disagree)\b", response) or re.search(
            r"\b(dis|not|no)[-]{0,1}\s*(agree|satisf)", response
        ):
            return -2 if (scale == 5 and strong) else -1

        # Agree / Satisfied (1, or +2 when strong on the 5-point scale)
        if re.search(r"\bagree\b", response) or re.search(r"satisf", response):
            return 2 if (scale == 5 and strong) else 1

        # Unable to classify
        return None

    conversion_summary = defaultdict(int)
    unconverted_phrases = set()

    if custom_mapping is None:
        mapping_func = default_mapping
        if debug:
            print(f"Using default {scale}-point mapping:")
            print("-1: Phrases containing 'disagree', 'do not agree', etc.")
            print(" 0: Phrases containing 'neutral', 'neither', 'unsure', etc.")
            print("+1: Phrases containing 'agree' (but not 'disagree' or 'not agree')")
            if scale == 5:
                print(
                    "±2: An intensifier (strongly/very/etc.) pushes agree/disagree"
                    " out to +2/-2"
                )
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
            f"{', '.join(unconverted_phrases)}"
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
                f"{', '.join(unconverted)}"
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
