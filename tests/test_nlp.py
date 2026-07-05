import numpy as np
import pandas as pd
import pytest

# Import the functions to test
from pandas_survey_toolkit.nlp import (
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


def test_encode_likert_scale_5(sample_df2):
    """scale=5 keeps agreement intensity (±2 for strongly/very variants)."""
    result = sample_df2.encode_likert(["Q1", "Q4"], scale=5, debug=False)

    # Strongly Agree -> 2, Disagree -> -1, Neutral -> 0, Agree -> 1,
    # Strongly Disagree -> -2
    assert list(result["likert_encoded_Q1"]) == [2, -1, 0, 1, -2]
    # Very Satisfied -> 2, neither... -> 0, dissatisfied -> -1,
    # very dis-satisfied -> -2, satisfied -> 1
    assert list(result["likert_encoded_Q4"]) == [2, 0, -1, -2, 1]


def test_encode_likert_scale_3_collapses_intensity(sample_df2):
    """scale=3 (default) collapses strong variants onto ±1."""
    result = sample_df2.encode_likert(["Q1"], scale=3, debug=False)
    assert list(result["likert_encoded_Q1"]) == [1, -1, 0, 1, -1]


def test_encode_likert_invalid_scale(sample_df2):
    with pytest.raises(ValueError):
        sample_df2.encode_likert(["Q1"], scale=4)


@pytest.fixture
def small_survey_df():
    """Few respondents (6) but many questions (10) with two opposite groups.

    This is the regime where UMAP+HDBSCAN struggles but respondent-correlation
    clustering excels.
    """
    agree_first = ["Agree"] * 5 + ["Disagree"] * 5
    disagree_first = ["Disagree"] * 5 + ["Agree"] * 5
    rows = [agree_first] * 3 + [disagree_first] * 3
    questions = [f"Q{i}" for i in range(1, 11)]
    df = pd.DataFrame(rows, columns=questions)
    df["respondent_id"] = range(len(df))
    return df, questions


def test_cluster_respondents_correlation_separates_groups(small_survey_df):
    df, questions = small_survey_df
    result = df.cluster_respondents_correlation(
        columns=questions, n_clusters=2, debug=False
    )

    assert "respondent_cluster_id" in result.columns
    labels = result["respondent_cluster_id"].tolist()
    # Two planted groups should land in two distinct, non-noise clusters.
    assert set(labels[:3]) == {labels[0]} and labels[0] != -1
    assert set(labels[3:6]) == {labels[3]} and labels[3] != -1
    assert labels[0] != labels[3]
    # Linkage is exposed for the dendrogram helper.
    assert "respondent_linkage" in result.attrs


def test_cluster_respondents_correlation_default_threshold(small_survey_df):
    df, questions = small_survey_df
    result = df.cluster_respondents_correlation(columns=questions, debug=False)
    # Default distance_threshold=1.0 separates positively- from
    # negatively-correlated respondents into (at least) two groups.
    non_noise = [c for c in result["respondent_cluster_id"].unique() if c != -1]
    assert len(non_noise) >= 2


def test_cluster_respondents_correlation_absolute_vs_signed():
    """'absolute' groups perfect opposites together; 'signed' splits them."""
    pro = ["Agree"] * 5 + ["Disagree"] * 5
    anti = ["Disagree"] * 5 + ["Agree"] * 5  # exact mirror of pro
    questions = [f"Q{i}" for i in range(1, 11)]
    df = pd.DataFrame([pro, pro, anti, anti], columns=questions)

    signed = df.cluster_respondents_correlation(
        columns=questions, n_clusters=2, distance="signed", debug=False
    )
    slabels = signed["respondent_cluster_id"].tolist()
    assert slabels[0] == slabels[1] and slabels[2] == slabels[3]
    assert slabels[0] != slabels[2]  # opposite camps split

    absolute = df.cluster_respondents_correlation(
        columns=questions, n_clusters=2, distance="absolute", debug=False
    )
    alabels = absolute["respondent_cluster_id"]
    # Mirror-image respondents are treated as identical -> all one bloc.
    assert alabels.nunique() == 1
    assert (alabels != -1).all()


def test_cluster_respondents_correlation_constant_respondent():
    """A respondent with no variation is left unclustered (-1) with a warning."""
    df = pd.DataFrame(
        {
            "Q1": ["Agree", "Disagree", "Agree"],
            "Q2": ["Agree", "Agree", "Disagree"],
            "Q3": ["Agree", "Disagree", "Agree"],
        }
    )
    with pytest.warns(UserWarning, match="no variation"):
        result = df.cluster_respondents_correlation(
            columns=["Q1", "Q2", "Q3"], n_clusters=2, debug=False
        )
    # Respondent 0 answered "Agree" to everything -> undefined correlation -> -1.
    assert result.loc[0, "respondent_cluster_id"] == -1


def test_cluster_respondents_correlation_conflicting_cut_args(small_survey_df):
    df, questions = small_survey_df
    with pytest.raises(ValueError):
        df.cluster_respondents_correlation(
            columns=questions, n_clusters=2, distance_threshold=0.5, debug=False
        )


def test_cluster_respondents_columns(sample_df):
    result = sample_df.cluster_respondents(columns=["Q1", "Q2", "Q3", "Q4"])
    assert "respondent_cluster_id" in result.columns
    assert "respondent_cluster_probability" in result.columns
    assert "likert_umap_x" in result.columns
    assert "likert_umap_y" in result.columns


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


def test_cluster_questions_constant_question():
    """A question everyone answers identically is left unclustered (-1)."""
    df = pd.DataFrame(
        {
            "Q1": ["Agree", "Agree", "Agree"],  # constant -> undefined correlation
            "Q2": ["Agree", "Disagree", "Agree"],
            "Q3": ["Disagree", "Agree", "Disagree"],
        }
    )
    with pytest.warns(UserWarning, match="no variation"):
        s = df.cluster_questions(columns=["Q1", "Q2", "Q3"], n_clusters=2, debug=False)
    assert s["Q1"] == -1


def test_cluster_survey_populates_attrs(sample_df):
    """cluster_survey clusters both axes and stashes plain-typed attrs."""
    result = sample_df.cluster_survey(columns=["Q1", "Q2", "Q3", "Q4"])
    assert "respondent_cluster_id" in result.columns
    assert result.attrs["respondent_method"] == "correlation"  # small survey -> auto
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


def test_cluster_respondents_small_sample_warns(small_survey_df):
    """Default min_cluster_size (20) exceeds the 6 respondents -> guard warning."""
    df, questions = small_survey_df
    with pytest.warns(UserWarning, match="exceeds the number of complete respondents"):
        df.cluster_respondents(columns=questions)
