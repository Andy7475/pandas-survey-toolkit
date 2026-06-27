"""
System tests for NLP text processing functions.
Tests run real models (no mocks) to verify end-to-end behaviour, including edge
cases such as all-NaN columns, empty strings and very short comments.
"""

import numpy as np
import pandas as pd
import pytest

from pandas_survey_toolkit.nlp import (
    remove_short_comments,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_df():
    return pd.DataFrame(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "I have 42 apples and <b>3</b> oranges.",
                "Short",
                np.nan,
                "Another sentence with some   extra   whitespace.",
            ]
        }
    )


@pytest.fixture(scope="module")
def spacy_df():
    """Run fit_spacy once per module — it loads a large model."""
    texts = [
        "The cats are running quickly through the park.",
        "Dogs were playing happily in the garden.",
        np.nan,
        "I am not going there again.",
        "She has been very satisfied with the results.",
    ]
    df = pd.DataFrame({"text": texts})
    return df.fit_spacy(input_column="text", output_column="spacy_output")


@pytest.fixture
def keywords_df():
    return pd.DataFrame(
        {
            "text": [
                "quick brown fox",
                "lazy dog runs",
                "fox and dog play",
                "quick fox jumps",
                "lazy brown dog",
                "fox runs fast",
            ],
            "keywords": [
                ["fox", "quick"],
                ["dog", "lazy"],
                ["fox", "dog"],
                ["quick", "fox"],
                ["dog", "lazy"],
                ["fox"],
            ],
        }
    )


# ---------------------------------------------------------------------------
# clean_survey_columns
# ---------------------------------------------------------------------------


class TestCleanSurveyColumns:
    def test_ms_forms_typical_columns(self):
        df = pd.DataFrame(
            columns=[
                "How satisfied are you? (1-5)",
                "Would you recommend us?",
                "Don't Know / Pass",
            ]
        )
        result = df.clean_survey_columns()
        assert list(result.columns) == [
            "how_satisfied_are_you_15",  # dash inside (1-5) is punctuation, removed
            "would_you_recommend_us",
            "dont_know_pass",
        ]

    def test_lowercase(self):
        df = pd.DataFrame(columns=["My Column", "ANOTHER COLUMN"])
        result = df.clean_survey_columns()
        assert result.columns[0] == "my_column"
        assert result.columns[1] == "another_column"

    def test_normalises_whitespace(self):
        df = pd.DataFrame(columns=["too  many   spaces"])
        result = df.clean_survey_columns()
        assert result.columns[0] == "too_many_spaces"

    def test_removes_punctuation(self):
        df = pd.DataFrame(columns=["Hello, World! How (are) you?"])
        result = df.clean_survey_columns()
        assert "," not in result.columns[0]
        assert "!" not in result.columns[0]
        assert "(" not in result.columns[0]
        assert "?" not in result.columns[0]

    def test_removes_apostrophe(self):
        df = pd.DataFrame(columns=["Don't know"])
        result = df.clean_survey_columns()
        assert "'" not in result.columns[0]
        assert result.columns[0] == "dont_know"

    def test_duplicate_names_after_cleaning_are_deduped(self):
        df = pd.DataFrame(columns=["How are you?", "How are you!"])
        result = df.clean_survey_columns()
        assert result.columns[0] == "how_are_you"
        assert result.columns[1] == "how_are_you_1"

    def test_already_clean_columns_unchanged(self):
        df = pd.DataFrame(columns=["q1", "q2", "score"])
        result = df.clean_survey_columns()
        assert list(result.columns) == ["q1", "q2", "score"]

    def test_data_preserved(self):
        df = pd.DataFrame({"How are you?": [1, 2, 3], "Score (1-5)": [4, 5, 6]})
        result = df.clean_survey_columns()
        assert list(result["how_are_you"]) == [1, 2, 3]
        # dash inside (1-5) is punctuation and gets removed
        assert list(result["score_15"]) == [4, 5, 6]

    def test_forward_slash_removed(self):
        df = pd.DataFrame(columns=["Yes / No", "Pass/Fail"])
        result = df.clean_survey_columns()
        assert "/" not in result.columns[0]
        assert "/" not in result.columns[1]


# ---------------------------------------------------------------------------
# preprocess_text
# ---------------------------------------------------------------------------


class TestPreprocessText:
    def test_defaults_strip_html_and_whitespace(self, text_df):
        result = text_df.preprocess_text(input_column="text", output_column="clean")
        assert "<b>" not in result["clean"][1]
        assert "  " not in result["clean"][4]

    def test_nan_preserved(self, text_df):
        result = text_df.preprocess_text(input_column="text", output_column="clean")
        assert pd.isna(result["clean"][3])
        assert result["clean"][0] == "The quick brown fox jumps over the lazy dog."

    def test_lowercase(self):
        df = pd.DataFrame({"text": ["Hello WORLD", "TEST Text"]})
        result = df.preprocess_text(
            input_column="text", output_column="clean", lower_case=True
        )
        assert result["clean"][0] == "hello world"
        assert result["clean"][1] == "test text"

    def test_remove_numbers(self):
        df = pd.DataFrame({"text": ["I have 42 apples", "No numbers here"]})
        result = df.preprocess_text(
            input_column="text", output_column="clean", remove_numbers=True
        )
        assert "42" not in result["clean"][0]
        assert "No" in result["clean"][1]

    def test_remove_stopwords(self):
        # gensim remove_stopwords is case-sensitive; use lowercase input
        df = pd.DataFrame({"text": ["the quick brown fox", "this is a test"]})
        result = df.preprocess_text(
            input_column="text",
            output_column="clean",
            remove_stopwords=True,
            remove_punctuation=False,
        )
        assert "the" not in result["clean"][0]
        assert "quick" in result["clean"][0]

    def test_max_comment_length_truncates(self):
        df = pd.DataFrame({"text": ["This is a long sentence that must be cut"]})
        result = df.preprocess_text(
            input_column="text", output_column="clean", max_comment_length=10
        )
        assert len(result["clean"][0]) <= 10
        assert "clean_was_truncated" in result.columns
        assert result["clean_was_truncated"][0] is True

    def test_max_comment_length_no_truncation_column_when_short(self):
        df = pd.DataFrame({"text": ["Hi"]})
        result = df.preprocess_text(
            input_column="text", output_column="clean", max_comment_length=100
        )
        assert "clean_was_truncated" in result.columns
        assert result["clean_was_truncated"][0] is False

    def test_flag_short_comments(self):
        df = pd.DataFrame({"text": ["Hi", "A longer response here"]})
        result = df.preprocess_text(
            input_column="text",
            output_column="clean",
            flag_short_comments=True,
            min_comment_length=5,
        )
        assert "clean_is_short" in result.columns
        assert result["clean_is_short"][0] is True
        assert result["clean_is_short"][1] is False

    def test_comment_length_column(self):
        df = pd.DataFrame({"text": ["Hello World"]})
        result = df.preprocess_text(
            input_column="text",
            output_column="clean",
            comment_length_column="char_count",
        )
        assert "char_count" in result.columns
        assert result["char_count"][0] > 0

    def test_no_output_column_overwrites_input(self):
        df = pd.DataFrame({"text": ["Hello  World"]})
        result = df.preprocess_text(input_column="text")
        assert result["text"][0] == "Hello World"

    def test_keep_sentence_punctuation_false(self):
        df = pd.DataFrame({"text": ["Hello, World! How are you?"]})
        result = df.preprocess_text(
            input_column="text",
            output_column="clean",
            keep_sentence_punctuation=False,
        )
        assert "," not in result["clean"][0]
        assert "!" not in result["clean"][0]

    def test_all_nan_column(self):
        df = pd.DataFrame({"text": [np.nan, np.nan, np.nan]})
        result = df.preprocess_text(input_column="text", output_column="clean")
        assert result["clean"].isna().all()

    def test_does_not_modify_original(self, text_df):
        original_values = text_df["text"].tolist()
        text_df.preprocess_text(input_column="text", output_column="clean")
        assert text_df["text"].tolist() == original_values


# ---------------------------------------------------------------------------
# remove_short_comments
# ---------------------------------------------------------------------------


class TestRemoveShortComments:
    def test_replaces_short_with_nan(self):
        df = pd.DataFrame({"text": ["Hi", "Hello World", "A longer sentence here"]})
        result = df.remove_short_comments(input_column="text", min_comment_length=5)
        assert pd.isna(result["text"][0])
        assert result["text"][1] == "Hello World"
        assert result["text"][2] == "A longer sentence here"

    def test_nan_stays_nan(self):
        df = pd.DataFrame({"text": [np.nan, "Hello World"]})
        result = df.remove_short_comments(input_column="text")
        assert pd.isna(result["text"][0])

    def test_exact_threshold_kept(self):
        df = pd.DataFrame({"text": ["Hello"]})  # exactly 5 chars
        result = df.remove_short_comments(input_column="text", min_comment_length=5)
        assert result["text"][0] == "Hello"

    def test_non_string_becomes_nan(self):
        df = pd.DataFrame({"text": [42, "Hello World"]})
        result = df.remove_short_comments(input_column="text", min_comment_length=5)
        assert pd.isna(result["text"][0])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"text": ["Hi", "Hello"]})
        remove_short_comments(df, input_column="text")
        assert df["text"][0] == "Hi"

    def test_all_short_all_nan(self):
        df = pd.DataFrame({"text": ["Hi", "No", "Ok"]})
        result = df.remove_short_comments(input_column="text", min_comment_length=5)
        assert result["text"].isna().all()


# ---------------------------------------------------------------------------
# fit_tfidf
# ---------------------------------------------------------------------------

TFIDF_TEXTS = [
    "the quick brown fox",
    "lazy dog runs fast",
    "fox and dog play together",
    "quick fox jumps high",
    "brown dog sits lazily",
    "the lazy fox sleeps",
    "fox chases the dog",
    "dog barks at the fox",
    "quick brown animal",
    "lazy sleeping animal",
]


class TestFitTfidf:
    def test_basic_keyword_extraction(self):
        df = pd.DataFrame({"text": TFIDF_TEXTS})
        result = df.fit_tfidf(
            input_column="text", output_column="keywords", top_n=2, threshold=0.1
        )
        assert "keywords" in result.columns
        assert isinstance(result["keywords"][0], list)
        assert len(result) == len(TFIDF_TEXTS)

    def test_nan_rows_preserved(self):
        texts = TFIDF_TEXTS[:5] + [np.nan] + TFIDF_TEXTS[5:]
        df = pd.DataFrame({"text": texts})
        result = df.fit_tfidf(
            input_column="text", output_column="keywords", threshold=0.1
        )
        assert pd.isna(result["keywords"][5])
        assert isinstance(result["keywords"][0], list)

    def test_append_features(self):
        df = pd.DataFrame({"text": TFIDF_TEXTS})
        result = df.fit_tfidf(
            input_column="text",
            output_column="keywords",
            append_features=True,
            threshold=0.1,
        )
        assert "keywords" in result.columns
        # Should have extra TF-IDF feature columns
        assert len(result.columns) > 2

    def test_bigrams(self):
        df = pd.DataFrame({"text": TFIDF_TEXTS})
        result = df.fit_tfidf(
            input_column="text",
            output_column="keywords",
            ngram_range=(1, 2),
            threshold=0.1,
        )
        assert "keywords" in result.columns

    def test_top_n_limits_keywords(self):
        df = pd.DataFrame({"text": TFIDF_TEXTS})
        result = df.fit_tfidf(
            input_column="text", output_column="keywords", top_n=1, threshold=0.0
        )
        for kws in result["keywords"].dropna():
            assert len(kws) <= 1

    def test_all_nan_input(self):
        df = pd.DataFrame({"text": [np.nan, np.nan]})
        result = df.fit_tfidf(input_column="text", output_column="keywords")
        assert "keywords" in result.columns
        assert result["keywords"].isna().all()


# ---------------------------------------------------------------------------
# get_lemma
# ---------------------------------------------------------------------------


class TestGetLemma:
    def test_basic_lemmatization(self, spacy_df):
        result = spacy_df.get_lemma(input_column="spacy_output", output_column="lemma")
        assert "lemma" in result.columns
        # "cats" should lemmatise to "cat", "running" to "run"
        assert "cat" in result["lemma"][0]
        assert "run" in result["lemma"][0]

    def test_nan_preserved(self, spacy_df):
        result = spacy_df.get_lemma(input_column="spacy_output", output_column="lemma")
        assert pd.isna(result["lemma"][2])
        assert isinstance(result["lemma"][0], str)

    def test_join_tokens_false_returns_list(self, spacy_df):
        result = spacy_df.get_lemma(
            input_column="spacy_output", output_column="lemma", join_tokens=False
        )
        assert isinstance(result["lemma"][0], list)

    def test_neg_dependency_kept_by_default(self, spacy_df):
        # "not" in "I am not going there again" has dep_=neg, kept by default
        result = spacy_df.get_lemma(input_column="spacy_output", output_column="lemma")
        assert "not" in result["lemma"][3]

    def test_keep_tokens(self, spacy_df):
        result = spacy_df.get_lemma(
            input_column="spacy_output",
            output_column="lemma",
            keep_tokens=["The"],
        )
        assert "lemma" in result.columns

    def test_keep_pos(self, spacy_df):
        result = spacy_df.get_lemma(
            input_column="spacy_output",
            output_column="lemma",
            keep_pos=["NOUN"],
        )
        assert "lemma" in result.columns

    def test_keep_dep_none(self, spacy_df):
        result = spacy_df.get_lemma(
            input_column="spacy_output",
            output_column="lemma",
            keep_dep=None,
        )
        assert "lemma" in result.columns

    def test_all_nan_input(self):
        df = pd.DataFrame({"spacy_output": [np.nan, np.nan]})
        result = df.get_lemma(input_column="spacy_output", output_column="lemma")
        assert result["lemma"].isna().all()


# ---------------------------------------------------------------------------
# refine_keywords
# ---------------------------------------------------------------------------


class TestRefineKeywords:
    def test_with_explicit_min_count(self, keywords_df):
        result = keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            min_count=2,
            debug=False,
        )
        assert "keywords" in result.columns
        assert isinstance(result["keywords"][0], list)

    def test_auto_min_count(self, keywords_df):
        result = keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            debug=False,
        )
        assert "keywords" in result.columns

    def test_custom_output_column_preserves_original(self, keywords_df):
        original_kws = keywords_df["keywords"].tolist()
        result = keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            output_column="refined",
            min_count=1,
            debug=False,
        )
        assert "refined" in result.columns
        assert "keywords" in result.columns
        # Original keyword column unchanged
        assert result["keywords"].tolist() == original_kws

    def test_debug_output_printed(self, keywords_df, capsys):
        keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            min_count=1,
            debug=True,
        )
        captured = capsys.readouterr()
        assert "Refinement complete" in captured.out
        assert "Min count used" in captured.out
        assert "unique keywords" in captured.out

    def test_no_debug_no_output(self, keywords_df, capsys):
        keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            min_count=1,
            debug=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_output_column_none_overwrites(self, keywords_df):
        result = keywords_df.refine_keywords(
            keyword_column="keywords",
            text_column="text",
            output_column=None,
            min_count=1,
            debug=False,
        )
        assert "keywords" in result.columns

    def test_non_list_keyword_value_returns_empty(self):
        """Defensive branch: a non-list (but non-NaN) keyword value produces []."""
        df = pd.DataFrame(
            {
                "text": ["quick brown fox", "lazy dog runs"],
                "keywords": ["fox", ["dog"]],  # first row is a string, not a list
            }
        )
        result = df.refine_keywords(
            keyword_column="keywords", text_column="text", min_count=1, debug=False
        )
        assert "keywords" in result.columns
        assert result["keywords"][0] == []


# ---------------------------------------------------------------------------
# cluster_comments  (full ML pipeline system test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster_comments_df():
    """Small but sufficient corpus for the full sentence-transformer + UMAP +
    HDBSCAN pipeline."""
    positive = [
        "I absolutely love this product",
        "Excellent quality and great service",
        "Outstanding experience every time",
        "Highly recommend to everyone",
        "Fantastic results, very happy",
        "Best purchase I have made",
    ] * 3
    negative = [
        "Terrible experience, very disappointed",
        "Poor quality, would not buy again",
        "Awful service and slow delivery",
        "Complete waste of money",
        "Very unhappy with this product",
        "Would not recommend to anyone",
    ] * 3
    return pd.DataFrame({"text": positive + negative})


def test_cluster_comments_output_columns(cluster_comments_df):
    result = cluster_comments_df.cluster_comments(
        input_column="text",
        min_cluster_size=3,
        n_neighbors=5,
    )
    assert "cluster" in result.columns
    assert "cluster_probability" in result.columns
    assert "sentence_embedding" in result.columns
    assert "umap_x" in result.columns
    assert "umap_y" in result.columns
    assert len(result) == len(cluster_comments_df)


def test_cluster_comments_nan_preserved(cluster_comments_df):
    df_with_nan = pd.concat(
        [cluster_comments_df, pd.DataFrame({"text": [np.nan, np.nan]})],
        ignore_index=True,
    )
    result = df_with_nan.cluster_comments(
        input_column="text", min_cluster_size=3, n_neighbors=5
    )
    assert pd.isna(result["cluster"]).sum() == 2


def test_cluster_comments_custom_output_columns(cluster_comments_df):
    result = cluster_comments_df.cluster_comments(
        input_column="text",
        output_columns=["group", "group_prob"],
        min_cluster_size=3,
        n_neighbors=5,
    )
    assert "group" in result.columns
    assert "group_prob" in result.columns


# ---------------------------------------------------------------------------
# extract_keywords  (full NLP pipeline system test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extract_keywords_df():
    """Representative corpus for the preprocess → spacy → lemma → tfidf →
    refine pipeline."""
    texts = [
        "Machine learning algorithms can process large datasets efficiently",
        "Deep learning models require substantial training data and computation",
        "Natural language processing helps computers understand human text",
        "Data science combines statistical methods with programming skills",
        "Python is widely used for data analysis and machine learning tasks",
        "Neural networks learn patterns from data through iterative training",
        "Text classification assigns documents to predefined category labels",
        "Feature engineering transforms raw data into useful model inputs",
        "Supervised learning uses labelled examples to train predictive models",
        "Unsupervised learning discovers hidden structure in unlabelled data",
        "Clustering algorithms group similar data points into coherent clusters",
        "Dimensionality reduction techniques simplify high-dimensional feature spaces",
    ] * 2
    return pd.DataFrame({"text": texts})


def test_extract_keywords_produces_output(extract_keywords_df):
    result = extract_keywords_df.extract_keywords(input_column="text")
    assert "keywords" in result.columns
    assert "refined_keywords" in result.columns
    assert "preprocessed_text" in result.columns
    assert "lemmatized_text" in result.columns


def test_extract_keywords_with_nan():
    texts = [
        "Machine learning is a powerful tool for data analysis",
        "Deep learning requires large amounts of training data",
        np.nan,
        "Natural language processing helps understand text",
        "Short",  # will be removed by remove_short_comments
    ] * 3
    df = pd.DataFrame({"text": texts})
    result = df.extract_keywords(input_column="text", min_df=1)
    assert "refined_keywords" in result.columns
    assert pd.isna(result["refined_keywords"][2])
