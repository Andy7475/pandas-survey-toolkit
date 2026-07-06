import warnings

import numpy as np
import pandas as pd
import pytest

# Import the functions to test
from pandas_survey_toolkit.nlp import (
    _check_encoded_range,
    fit_sentence_transformer,
    fit_spacy,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Q1": np.random.choice(
                ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
                100,
            ),
            "Q2": np.random.choice(
                ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
                100,
            ),
            "Q3": np.random.choice(
                ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
                100,
            ),
            "Q4": np.random.choice(
                ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
                100,
            ),
            "OtherColumn": np.random.rand(100),
        }
    )


def test_cluster_questions_pattern(sample_df):
    """cluster_questions accepts a regex pattern and returns a per-question Series."""
    s = sample_df.cluster_questions(pattern="^Q", n_clusters=2, debug=False)
    assert isinstance(s, pd.Series)
    assert set(s.index) == {"Q1", "Q2", "Q3", "Q4"}


def test_cluster_questions_error_no_columns_or_pattern(sample_df):
    with pytest.raises(ValueError):
        sample_df.cluster_questions()


# Test for fit_sentence_transformer
def test_fit_sentence_transformer():
    pytest.importorskip(
        "sentence_transformers", reason="needs the optional 'nlp' extra"
    )
    df = pd.DataFrame(
        {
            "text": ["Hello world", "Test sentence", np.nan, "Another test"],
        }
    )

    result = fit_sentence_transformer(df, input_column="text")

    assert "sentence_embedding" in result.columns
    assert len(result) == 4
    assert isinstance(result["sentence_embedding"][0], np.ndarray)
    assert np.isnan(result["sentence_embedding"][2]).all()  # Check if NaN is preserved


def test_extract_sentiment():
    pytest.importorskip("transformers", reason="needs the optional 'nlp' extra")
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "text": [
                "This is amazing!",
                "I hate this, it's awful",
                "Neutral statement",
                np.nan,
                "Another positive example",
            ]
        }
    )

    # Apply the sentiment analysis
    result = df.extract_sentiment(input_column="text")

    # Check if new columns are added
    assert "positive" in result.columns
    assert "neutral" in result.columns
    assert "negative" in result.columns
    assert "sentiment" in result.columns

    # Check if the DataFrame has the correct shape
    assert result.shape == (5, 5)  # Original column + 4 new columns

    # Check positive sentiment
    assert result.loc[0, "sentiment"] == "positive"
    assert result.loc[0, "positive"] > result.loc[0, "negative"]

    # Check negative sentiment
    assert result.loc[1, "sentiment"] == "negative"
    assert result.loc[1, "negative"] > result.loc[1, "positive"]

    # Check if NaN is preserved
    assert np.isnan(result.loc[3, "positive"])
    assert np.isnan(result.loc[3, "neutral"])
    assert np.isnan(result.loc[3, "negative"])
    assert pd.isna(result.loc[3, "sentiment"])

    # Check if all sentiment scores are between 0 and 1
    assert (
        (result["positive"] >= 0) & (result["positive"] <= 1)
        | result["positive"].isna()
    ).all()
    assert (
        (result["neutral"] >= 0) & (result["neutral"] <= 1) | result["neutral"].isna()
    ).all()
    assert (
        (result["negative"] >= 0) & (result["negative"] <= 1)
        | result["negative"].isna()
    ).all()

    # Check if sentiment labels are correct
    assert set(result["sentiment"].dropna().unique()) <= {
        "positive",
        "neutral",
        "negative",
    }


def test_fit_spacy():
    pytest.importorskip("spacy", reason="needs the optional 'nlp' extra")
    import spacy

    # Create a sample DataFrame
    df = pd.DataFrame(
        {"comments": ["This is a test", "Another comment", np.nan, "SpaCy is cool"]}
    )

    # Apply the fit_spacy function
    result = fit_spacy(df, input_column="comments")

    # Check if the new column exists
    assert "spacy_output" in result.columns

    # Check if the number of rows is preserved
    assert len(result) == len(df)

    # Check if the entries are spaCy Doc objects (where not NaN)
    for i, row in result.iterrows():
        if pd.notna(row["comments"]):
            assert isinstance(row["spacy_output"], spacy.tokens.doc.Doc)
        else:
            assert pd.isna(row["spacy_output"])

    # Check if the content of the spaCy Doc objects matches the input
    nlp = spacy.load("en_core_web_md")
    for i, row in result.iterrows():
        if pd.notna(row["comments"]):
            assert row["spacy_output"].text == row["comments"]
            assert row["spacy_output"].text == nlp(row["comments"]).text


@pytest.fixture
def sample_df2():
    return pd.DataFrame(
        {
            "Q1": [
                "Strongly Agree",
                "Disagree",
                "Neither Agree nor Disagree",
                "Agree",
                "Strongly Disagree",
            ],
            "Q2": ["Agree", "Disagree", "Neutral", "Strongly Agree", "Do not agree"],
            "Q3": ["Strongly Agree", np.nan, "Neutral", "Agree", "Unconverted"],
            "Q4": [
                "Very Satisfied",
                "neither satisfied nor dissatisfied",
                "dissatisfied",
                "very dis-satisfied",
                "satisfied",
            ],
        }
    )


@pytest.fixture
def custom_mapping():
    return {
        "strongly agree": 2,
        "agree": 1,
        "neither agree nor disagree": 0,
        "neutral": 0,
        "disagree": -1,
        "strongly disagree": -2,
        "do not agree": -1,
    }


def test_default_mapping(sample_df2):
    result = sample_df2.encode_likert(["Q1", "Q2", "Q4"])

    expected_Q1 = [1, -1, 0, 1, -1]
    expected_Q2 = [1, -1, 0, 1, -1]
    expected_Q4 = [1, 0, -1, -1, 1]

    assert list(result["likert_encoded_Q1"]) == expected_Q1
    assert list(result["likert_encoded_Q2"]) == expected_Q2
    assert list(result["likert_encoded_Q4"]) == expected_Q4


def test_column_production(sample_df2):
    result = sample_df2.encode_likert(["Q1", "Q2", "Q3"])

    expected_columns = set(sample_df2.columns) | {
        "likert_encoded_Q1",
        "likert_encoded_Q2",
        "likert_encoded_Q3",
    }
    assert set(result.columns) == expected_columns


def test_nan_handling(sample_df2):
    result = sample_df2.encode_likert(["Q3"])

    assert pd.isna(result.loc[1, "likert_encoded_Q3"])
    assert result.loc[0, "likert_encoded_Q3"] == 1  # 'Strongly Agree'
    assert result.loc[2, "likert_encoded_Q3"] == 0  # 'Neutral'


def test_custom_mapping(sample_df2, custom_mapping):
    result = sample_df2.encode_likert(["Q1", "Q2"], custom_mapping=custom_mapping)

    expected_Q1 = [2, -1, 0, 1, -2]
    expected_Q2 = [1, -1, 0, 2, -1]

    assert list(result["likert_encoded_Q1"]) == expected_Q1
    assert list(result["likert_encoded_Q2"]) == expected_Q2


def test_custom_mapping_nan_handling(sample_df2, custom_mapping):
    result = sample_df2.encode_likert(["Q3"], custom_mapping=custom_mapping)

    assert pd.isna(result.loc[1, "likert_encoded_Q3"])
    assert result.loc[0, "likert_encoded_Q3"] == 2  # 'Strongly Agree'
    assert result.loc[2, "likert_encoded_Q3"] == 0  # 'Neutral'


def test_output_prefix(sample_df2):
    result = sample_df2.encode_likert(["Q1"], output_prefix="custom_")

    assert "custom_Q1" in result.columns
    assert "likert_encoded_Q1" not in result.columns


def test_unconverted_warning(sample_df2, custom_mapping):
    with pytest.warns(UserWarning, match="The following phrases were not converted"):
        sample_df2.encode_likert(["Q3"], custom_mapping=custom_mapping)


def test_default_mapping_warning(sample_df2):
    with pytest.warns(
        UserWarning, match="The default mapping didn't convert the following responses"
    ):
        sample_df2.encode_likert(["Q3"])


def test_dont_know_with_slash():
    """Regression test: "Don't Know / Pass" contains apostrophe and slash."""
    df = pd.DataFrame(
        {
            "Q1": ["Don't Know / Pass", "Don't Know", "Agree", "Disagree", "Neutral"],
        }
    )
    result = df.encode_likert(["Q1"], debug=False)
    assert result.loc[0, "likert_encoded_Q1"] == 0, (
        '"Don\'t Know / Pass" should map to 0'
    )
    assert result.loc[1, "likert_encoded_Q1"] == 0, '"Don\'t Know" should map to 0'
    assert result.loc[2, "likert_encoded_Q1"] == 1
    assert result.loc[3, "likert_encoded_Q1"] == -1
    assert result.loc[4, "likert_encoded_Q1"] == 0


@pytest.fixture
def small_survey_df():
    """Few respondents (6) but many questions (10) with two opposite groups.

    This is the regime where UMAP+HDBSCAN struggles but cosine + hierarchical
    clustering excels.
    """
    agree_first = ["Agree"] * 5 + ["Disagree"] * 5
    disagree_first = ["Disagree"] * 5 + ["Agree"] * 5
    rows = [agree_first] * 3 + [disagree_first] * 3
    questions = [f"Q{i}" for i in range(1, 11)]
    df = pd.DataFrame(rows, columns=questions)
    df["respondent_id"] = range(len(df))
    return df, questions


def test_cluster_respondents_cosine_separates_groups(small_survey_df):
    df, questions = small_survey_df
    result = df.cluster_respondents_cosine(columns=questions, n_clusters=2, debug=False)

    assert "respondent_cluster_id" in result.columns
    labels = result["respondent_cluster_id"].tolist()
    # Two planted groups should land in two distinct, non-noise clusters.
    assert set(labels[:3]) == {labels[0]} and labels[0] != -1
    assert set(labels[3:6]) == {labels[3]} and labels[3] != -1
    assert labels[0] != labels[3]
    # Linkage is exposed for the dendrogram helper.
    assert "respondent_linkage" in result.attrs


def test_cluster_respondents_cosine_separates_agreers_from_disagreers():
    """Cosine clusters all-agree and all-disagree respondents into two distinct
    groups - Pearson could not (they are zero-variance and undefined)."""
    q = [f"Q{i}" for i in range(1, 7)]
    df = pd.DataFrame([["Agree"] * 6] * 4 + [["Disagree"] * 6] * 4, columns=q)
    labels = df.cluster_respondents_cosine(columns=q, n_clusters=2, debug=False)[
        "respondent_cluster_id"
    ].tolist()
    assert labels[:4] == [labels[0]] * 4 and labels[0] != -1
    assert labels[4:] == [labels[4]] * 4 and labels[4] != -1
    assert labels[0] != labels[4]


def test_cluster_respondents_cosine_default_threshold(small_survey_df):
    df, questions = small_survey_df
    result = df.cluster_respondents_cosine(columns=questions, debug=False)
    # Default cosine distance_threshold=1.0 splits opposite-direction respondents
    # into (at least) two groups.
    non_noise = [c for c in result["respondent_cluster_id"].unique() if c != -1]
    assert len(non_noise) >= 2


def test_cluster_respondents_cosine_all_neutral():
    """An all-neutral (zero) respondent has no direction -> unclustered (-1)."""
    df = pd.DataFrame(
        {
            "Q1": ["Neutral", "Disagree", "Agree"],
            "Q2": ["Neutral", "Agree", "Disagree"],
            "Q3": ["Neutral", "Disagree", "Agree"],
        }
    )
    with pytest.warns(UserWarning, match="all-neutral"):
        result = df.cluster_respondents_cosine(
            columns=["Q1", "Q2", "Q3"], n_clusters=2, debug=False
        )
    # Respondent 0 answered everything neutral -> zero vector -> -1.
    assert result.loc[0, "respondent_cluster_id"] == -1


def test_cluster_respondents_cosine_conflicting_cut_args(small_survey_df):
    df, questions = small_survey_df
    with pytest.raises(ValueError):
        df.cluster_respondents_cosine(
            columns=questions, n_clusters=2, distance_threshold=0.5, debug=False
        )


def test_cluster_respondents_auto_uses_cosine(sample_df):
    """The dispatcher picks cosine for a small/medium survey (no UMAP coords)."""
    result = sample_df.cluster_respondents(columns=["Q1", "Q2", "Q3", "Q4"])
    assert "respondent_cluster_id" in result.columns
    assert "likert_umap_x" not in result.columns  # cosine path, not UMAP


def test_cluster_respondents_umap_method(sample_df):
    """method='umap' runs the UMAP path (embeds coordinates + probability)."""
    result = sample_df.cluster_respondents(
        columns=["Q1", "Q2", "Q3", "Q4"], method="umap"
    )
    assert "respondent_cluster_id" in result.columns
    assert "respondent_cluster_probability" in result.columns
    assert "likert_umap_x" in result.columns
    assert "likert_umap_y" in result.columns


def test_cluster_respondents_invalid_method(sample_df):
    with pytest.raises(ValueError):
        sample_df.cluster_respondents(columns=["Q1", "Q2"], method="banana")


@pytest.fixture
def question_groups_df():
    """Q1==Q2 follow one response pattern, Q3==Q4 follow an uncorrelated one."""
    p1 = [
        "Agree",
        "Disagree",
        "Agree",
        "Agree",
        "Disagree",
        "Agree",
        "Disagree",
        "Disagree",
        "Agree",
        "Disagree",
    ]
    p2 = [
        "Disagree",
        "Disagree",
        "Agree",
        "Disagree",
        "Agree",
        "Agree",
        "Agree",
        "Disagree",
        "Disagree",
        "Agree",
    ]
    return pd.DataFrame({"Q1": p1, "Q2": p1, "Q3": p2, "Q4": p2})


def test_cluster_questions_returns_series(question_groups_df):
    """cluster_questions returns a per-question Series that groups co-answered Qs."""
    s = question_groups_df.cluster_questions(
        columns=["Q1", "Q2", "Q3", "Q4"], n_clusters=2, debug=False
    )
    assert isinstance(s, pd.Series)
    assert list(s.index) == ["Q1", "Q2", "Q3", "Q4"]  # original names
    assert s["Q1"] == s["Q2"]  # identical questions cluster together
    assert s["Q3"] == s["Q4"]
    assert s["Q1"] != s["Q3"]  # the two groups are distinct
    # Ordering + linkage exposed on attrs; Series is CSV-able for other analysis.
    assert set(s.attrs["question_order"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert isinstance(s.to_csv(), str)


def test_cluster_questions_all_neutral_question():
    """A question everyone answers neutral is a zero vector -> unclustered (-1)."""
    df = pd.DataFrame(
        {
            "Q1": ["Neutral", "Neutral", "Neutral"],  # zero vector, no direction
            "Q2": ["Agree", "Disagree", "Agree"],
            "Q3": ["Disagree", "Agree", "Disagree"],
        }
    )
    with pytest.warns(UserWarning, match="all-neutral"):
        s = df.cluster_questions(columns=["Q1", "Q2", "Q3"], n_clusters=2, debug=False)
    assert s["Q1"] == -1


def test_cluster_survey_populates_attrs(sample_df):
    """cluster_survey clusters both axes and stashes plain-typed attrs."""
    result = sample_df.cluster_survey(columns=["Q1", "Q2", "Q3", "Q4"])
    assert "respondent_cluster_id" in result.columns
    assert result.attrs["respondent_method"] == "cosine"  # small survey -> auto
    # attrs are all plain (list/dict/scalar) so pandas ops still work.
    assert isinstance(result.attrs["question_cluster_id"], dict)
    assert isinstance(result.attrs["question_order"], list)
    # a groupby (which triggers pandas attrs propagation) must not raise
    result.groupby("respondent_cluster_id").size()


def test_cluster_survey_umap_warns_on_small(small_survey_df):
    """Forcing umap on a tiny survey warns (illogical choice for the size)."""
    df, questions = small_survey_df
    with pytest.warns(UserWarning, match="unreliable with only"):
        df.cluster_survey(columns=questions, respondent_method="umap")


def test_cluster_respondents_umap_small_sample_warns(small_survey_df):
    """The UMAP path shrinks min_cluster_size (with a warning) on a tiny survey."""
    df, questions = small_survey_df
    with pytest.warns(UserWarning, match="exceeds the number of complete respondents"):
        df.cluster_respondents_umap(columns=questions)


def test_cluster_questions_nan_default_fills(question_groups_df):
    """Default nan_strategy="fill" treats a missing answer as neutral and warns."""
    df = question_groups_df.copy()
    df.loc[0, "Q1"] = np.nan

    with pytest.warns(UserWarning, match="missing Likert answer"):
        s = df.cluster_questions(
            columns=["Q1", "Q2", "Q3", "Q4"], n_clusters=2, debug=False
        )
    assert isinstance(s, pd.Series)
    assert set(s.index) == {"Q1", "Q2", "Q3", "Q4"}


def test_cluster_questions_nan_ignore_drops_row(question_groups_df):
    """nan_strategy="ignore" drops the affected respondent from the computation."""
    df = question_groups_df.copy()
    df.loc[0, "Q1"] = np.nan

    with pytest.warns(UserWarning, match="excluded from clustering"):
        s = df.cluster_questions(
            columns=["Q1", "Q2", "Q3", "Q4"],
            n_clusters=2,
            debug=False,
            nan_strategy="ignore",
        )

    baseline = question_groups_df.drop(index=0).cluster_questions(
        columns=["Q1", "Q2", "Q3", "Q4"], n_clusters=2, debug=False
    )
    assert s.to_dict() == baseline.to_dict()


def test_cluster_questions_invalid_nan_strategy(question_groups_df):
    with pytest.raises(ValueError, match="nan_strategy"):
        question_groups_df.cluster_questions(
            columns=["Q1", "Q2", "Q3", "Q4"], nan_strategy="bogus", debug=False
        )


def test_cluster_respondents_cosine_nan_strategy_fill_vs_ignore(small_survey_df):
    """ "ignore" excludes the incomplete respondent (-1); "fill" clusters it."""
    df, questions = small_survey_df
    df = df.copy()
    df.loc[0, "Q1"] = np.nan

    with pytest.warns(UserWarning, match="excluded from clustering"):
        result_ignore = df.cluster_respondents_cosine(
            columns=questions, n_clusters=2, debug=False, nan_strategy="ignore"
        )
    assert result_ignore.loc[0, "respondent_cluster_id"] == -1

    with pytest.warns(UserWarning, match="treated as neutral"):
        result_fill = df.cluster_respondents_cosine(
            columns=questions, n_clusters=2, debug=False, nan_strategy="fill"
        )
    assert result_fill.loc[0, "respondent_cluster_id"] != -1


def test_cluster_respondents_cosine_invalid_nan_strategy(small_survey_df):
    df, questions = small_survey_df
    with pytest.raises(ValueError, match="nan_strategy"):
        df.cluster_respondents_cosine(
            columns=questions, nan_strategy="bogus", debug=False
        )


def test_cluster_respondents_umap_nan_strategy_fill_vs_ignore(sample_df):
    """ "ignore" leaves the incomplete respondent's outputs NaN; "fill" clusters it."""
    df = sample_df.copy()
    df.loc[0, "Q1"] = np.nan
    questions = ["Q1", "Q2", "Q3", "Q4"]

    result_ignore = df.cluster_respondents_umap(
        columns=questions, nan_strategy="ignore"
    )
    assert pd.isna(result_ignore.loc[0, "respondent_cluster_id"])

    result_fill = df.cluster_respondents_umap(columns=questions, nan_strategy="fill")
    assert pd.notna(result_fill.loc[0, "respondent_cluster_id"])


def test_cluster_respondents_umap_invalid_nan_strategy(sample_df):
    with pytest.raises(ValueError, match="nan_strategy"):
        sample_df.cluster_respondents_umap(
            columns=["Q1", "Q2", "Q3", "Q4"], nan_strategy="bogus"
        )


def test_cosine_cluster_rejects_ward_linkage(small_survey_df):
    """ward/centroid/median assume Euclidean distances; reject them for cosine."""
    df, questions = small_survey_df
    for bad_method in ("ward", "centroid", "median"):
        with pytest.raises(ValueError, match="linkage_method must be one of"):
            df.cluster_respondents_cosine(
                columns=questions, linkage_method=bad_method, debug=False
            )

    for ok_method in ("single", "complete", "average", "weighted"):
        result = df.cluster_respondents_cosine(
            columns=questions, linkage_method=ok_method, n_clusters=2, debug=False
        )
        assert "respondent_cluster_id" in result.columns


def test_check_encoded_range_warns_on_stray_values():
    """A value outside the mapping's scale (besides NaN) should warn."""
    df = pd.DataFrame({"likert_encoded_Q1": [-1, 0, 1, 5, np.nan]})
    with pytest.warns(UserWarning, match="values outside the expected"):
        _check_encoded_range(df, ["likert_encoded_Q1"], likert_mapping=None)


def test_check_encoded_range_silent_for_expected_values():
    """Values within the default (or custom) scale, plus NaN, should not warn."""
    df = pd.DataFrame({"likert_encoded_Q1": [-1, 0, 1, np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_encoded_range(df, ["likert_encoded_Q1"], likert_mapping=None)

    custom_mapping = {"strongly agree": 2, "agree": 1, "disagree": -2}
    df2 = pd.DataFrame({"likert_encoded_Q1": [-2, 1, 2, np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_encoded_range(df2, ["likert_encoded_Q1"], likert_mapping=custom_mapping)
