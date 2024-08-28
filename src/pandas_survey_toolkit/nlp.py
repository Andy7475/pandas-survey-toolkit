import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from gensim.parsing.preprocessing import (remove_stopwords,
                                          strip_multiple_whitespaces,
                                          strip_numeric, strip_tags)
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pandas_survey_toolkit.analytics import fit_cluster_hdbscan, fit_umap
from pandas_survey_toolkit.utils import (apply_vectorizer, combine_results,
                                         create_masked_df)


@pf.register_dataframe_method
def fit_sentence_transformer(df, input_column:str, model_name='all-MiniLM-L6-v2', output_column="sentence_embedding"):
    """Adds a list of vector embeddings for each string in the input column. These can then be used for downstream
    tasks like clustering"""
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
def extract_sentiment(df, input_column: str, output_columns=["positive", "neutral", "negative", "sentiment"]):
    """
    Extract sentiment from text using the cardiffnlp/twitter-roberta-base-sentiment model.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to analyze.
    output_columns (list): List of column names for the output. Default is ["positive", "neutral", "negative", "sentiment"].
    
    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for sentiment scores and labels.
    """
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    masked_df, mask = create_masked_df(df, [input_column])
    
    def analyze_sentiment(text):
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        return scores
    
    sentiment_scores = masked_df[input_column].apply(analyze_sentiment)
    
    masked_df[output_columns[0]] = sentiment_scores.apply(lambda x: x[2])  # Positive
    masked_df[output_columns[1]] = sentiment_scores.apply(lambda x: x[1])  # Neutral
    masked_df[output_columns[2]] = sentiment_scores.apply(lambda x: x[0])  # Negative
    
    masked_df[output_columns[3]] = masked_df[[output_columns[0], output_columns[1], output_columns[2]]].idxmax(axis=1)
    masked_df[output_columns[3]] = masked_df[output_columns[3]].map({output_columns[0]: 'positive', 
                                                                     output_columns[1]: 'neutral', 
                                                                     output_columns[2]: 'negative'})
    
    df_to_return = combine_results(df, masked_df, mask, output_columns)
    return df_to_return

@pf.register_dataframe_method
def cluster_comments(df:pd.DataFrame, input_column:str, output_columns:str=["cluster", "cluster_probability"]):
    """applies a pipeline of 1) vector embeddings 2) dimensional reduction 3) clustering
    to assign each row a cluster ID so that similar free text comments (found in the input_column) can be grouped together.
    Returns a modified dataframe. If you want control over parameters for the various functions,
    then apply them separately. The defaults should be OK in most cases."""

    df_temp = (df.fit_sentence_transformer(input_column = input_column,
                                                 output_column="sentence_embedding")
                    .fit_umap(input_columns="sentence_embedding",
                              embeddings_in_list=True)
                    .fit_cluster_hdbscan(output_columns=output_columns))
    
    return df_temp

@pf.register_dataframe_method
def fit_tfidf(df: pd.DataFrame, 
              input_column: str, 
              output_column: str = 'keywords', 
              top_n: int = 3, 
              threshold: float = 0.0, 
              append_features: bool = False,
              ngram_range: Tuple[int, int] = (1, 1),
              **tfidf_kwargs) -> pd.DataFrame:
    """
    Apply TF-IDF vectorization to a text column and extract top N keywords for each document,
    while preserving NaN values in the original DataFrame. Supports n-gram extraction.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to vectorize.
    output_column (str): Name of the column to store the extracted keywords. Default is 'keywords'.
    top_n (int): Number of top keywords to extract for each document. Default is 5.
    threshold (float): Minimum TF-IDF score for a keyword to be included. Default is 0.0.
    append_features (bool): If True, append all TF-IDF features to the DataFrame. Default is False.
    ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
                         Default is (1, 1) which means only unigrams. Set to (1, 2) for unigrams and bigrams, and so on.
    **tfidf_kwargs: Additional keyword arguments to pass to TfidfVectorizer.

    Returns:
    pandas.DataFrame: The input DataFrame with an additional column containing the top keywords.
    """
    # Create a masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])

    # Ensure ngram_range is included in the TfidfVectorizer parameters
    tfidf_kwargs['ngram_range'] = ngram_range

    # Apply TF-IDF vectorization to the masked DataFrame
    tfidf_features, _, feature_names = apply_vectorizer(masked_df, input_column, vectorizer_name='TfidfVectorizer', **tfidf_kwargs)
    
    def extract_top_keywords(row: pd.Series) -> List[str]:
        # Get indices of top N TF-IDF scores
        top_indices = row.nlargest(top_n).index
        
        # Filter based on threshold and get the corresponding feature names
        top_keywords = [feature_names[i] for i, idx in enumerate(tfidf_features.columns) if idx in top_indices and row[idx] >= threshold]
        
        # Sort keywords based on their order in the original text
        original_text = masked_df.loc[row.name, input_column]
        return sorted(top_keywords, key=lambda x: original_text.lower().index(x.lower()) 
                      if x.lower() in original_text.lower() else len(original_text))

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

import pandas as pd
import pandas_flavor as pf
import spacy

from pandas_survey_toolkit.utils import combine_results, create_masked_df


@pf.register_dataframe_method
def fit_spacy(df, input_column: str, output_column: str = "spacy_output"):
    """
    Apply the en_core_web_md spaCy model to the specified column of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to analyze.
    output_column (str): Name of the output column. Default is "spacy_output".
    
    Returns:
    pandas.DataFrame: The input DataFrame with an additional column containing spaCy doc objects.
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
def get_lemma(df: pd.DataFrame, 
              input_column: str = 'spacy_output', 
              output_column: str = 'lemmatized_text', 
              text_pos: List[str] = ['PRON'],
              remove_punct: bool = True,
              remove_space: bool = True,
              remove_stop: bool = True,
              keep_tokens: Union[List[str], None] = None,
              keep_pos: Union[List[str], None] = None,
              keep_dep: Union[List[str], None] = None,
              join_tokens: bool = True) -> pd.DataFrame:
    """
    Extract lemmatized text from the spaCy doc objects in the specified column.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing spaCy doc objects. Default is 'spacy_output'.
    output_column (str): Name of the output column for lemmatized text. Default is 'lemmatized_text'.
    text_pos (List[str]): List of POS tags to exclude from lemmatization and return the text. Default is ['PRON'].
    remove_punct (bool): Whether to remove punctuation. Default is True.
    remove_space (bool): Whether to remove whitespace tokens. Default is True.
    remove_stop (bool): Whether to remove stop words. Default is True.
    keep_tokens (List[str]): List of token texts to always keep. Default is None.
    keep_pos (List[str]): List of POS tags to always keep. Default is None.
    keep_dep (List[str]): List of dependency labels to always keep. Default is None.
    join_tokens (bool): Whether to join tokens into a string. If False, returns a list of tokens. Default is True.
    
    Returns:
    pandas.DataFrame: The input DataFrame with an additional column containing lemmatized text or token list.
    """
    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])
    
    def remove_token(token):
        """
        Returns True if the token should be removed.
        """
        if (keep_tokens and token.text in keep_tokens) or \
           (keep_pos and token.pos_ in keep_pos) or \
           (keep_dep and token.dep_ in keep_dep):
            return False
        return ((remove_punct and token.is_punct) or
                (remove_space and token.is_space) or
                (remove_stop and token.is_stop))

    def process_text(doc):
        tokens = [token.text if token.pos_ in text_pos else token.lemma_ 
                  for token in doc if not remove_token(token)]
        return ' '.join(tokens) if join_tokens else tokens
    
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
    normalize_whitespace: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    flag_short_comments: bool = False,
    min_comment_length: int = 5,
    max_comment_length: int = None,
    remove_extra_punctuation: bool = True,
    keep_sentence_punctuation: bool = True,
    comment_length_column: str = None
) -> pd.DataFrame:
    """
    Preprocess text data in the specified column, tailored for survey responses.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to preprocess.
    output_column (str): Name of the output column. If None, overwrites the input column.
    remove_html (bool): Whether to remove unexpected HTML tags. Default is True.
    normalize_whitespace (bool): Whether to normalize whitespace. Default is True.
    remove_numbers (bool): Whether to remove numbers. Default is False.
    remove_stopwords (bool): Whether to remove stop words. Default is False.
    flag_short_comments (bool): Whether to flag very short comments. Default is False.
    min_comment_length (int): Minimum length of comment to not be flagged as short. Default is 5.
    max_comment_length (int): Maximum length of comment to keep. If None, keeps full length. Default is None.
    remove_extra_punctuation (bool): Whether to remove extra punctuation. Default is True.
    keep_sentence_punctuation (bool): Whether to keep sentence-level punctuation. Default is True.
    comment_length_column (str): Name of the column to store comment lengths. If None, no column is added. Default is None.
    
    Returns:
    pandas.DataFrame: The input DataFrame with preprocessed text and optionally new columns for short comments, truncation info, and comment length.
    """
    output_column = output_column or input_column
    
    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [input_column])
    
    def process_text(text):
        if remove_html:
            text = strip_tags(text)
        
        if normalize_whitespace:
            text = strip_multiple_whitespaces(text)
        
        if remove_numbers:
            text = strip_numeric(text)
        
        if remove_stopwords:
            text = remove_stopwords(text)
        
        if remove_extra_punctuation:
            if keep_sentence_punctuation:
                # Keep commas, periods, question marks, exclamation points, quotation marks, and apostrophes
                text = re.sub(r"([^.,!?\"'\s])\1+", r"\1", text)
                # Remove spaces before punctuation, but not before apostrophes
                text = re.sub(r"\s([.,!?\"](?:\s|$))", r"\1", text)
                # Ensure we don't have multiple apostrophes
                text = re.sub(r"'{2,}", "'", text)
                # Ensure apostrophes are used correctly (e.g., don't remove apostrophe in "don't")
                text = re.sub(r"\s'|'\s", " ", text)  # Remove single quotes if they're by themselves
            else:
                # Keep apostrophes, remove other repeated punctuation
                text = re.sub(r"([^\w\s'])\1+", r"\1", text)
                
        
        text = text.strip()
        
        if max_comment_length:
            text = text[:max_comment_length]
        
        return text
    
    # Apply processing
    masked_df[output_column] = masked_df[input_column].apply(process_text)
    
    columns_to_combine = [output_column]
    
    if flag_short_comments:
        short_comment_col = f"{output_column}_is_short"
        masked_df[short_comment_col] = masked_df[output_column].str.len() < min_comment_length
        columns_to_combine.append(short_comment_col)
    
    if max_comment_length:
        truncated_col = f"{output_column}_was_truncated"
        masked_df[truncated_col] = masked_df[input_column].str.len() > max_comment_length
        columns_to_combine.append(truncated_col)
    
    if comment_length_column:
        masked_df[comment_length_column] = masked_df[output_column].str.len()
        columns_to_combine.append(comment_length_column)
    
    # Combine results
    df_to_return = combine_results(df, masked_df, mask, columns_to_combine)
    
    return df_to_return