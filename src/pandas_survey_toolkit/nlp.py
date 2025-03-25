import re
import warnings
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
import spacy
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
def cluster_questions(df, columns=None, pattern=None, likert_mapping=None, 
                      umap_n_neighbors=15, umap_min_dist=0.1,
                      hdbscan_min_cluster_size=20, hdbscan_min_samples=None,
                      cluster_selection_epsilon=0.4):
    """
    Cluster Likert scale questions based on response patterns.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): List of column names to cluster. If None, all columns matching the pattern will be used.
    pattern (str): Regex pattern to match column names. Used if columns is None.
    likert_mapping (dict): Custom mapping for Likert scale responses. If None, default mapping is used.
    umap_n_neighbors (int): The size of local neighborhood for UMAP. Default is 15.
    umap_min_dist (float): The minimum distance between points in UMAP. Default is 0.1.
    umap_n_components (int): The number of dimensions for UMAP output. Default is 2.
    hdbscan_min_cluster_size (int): The minimum size of clusters for HDBSCAN. Default is 5.
    hdbscan_min_samples (int): The number of samples in a neighborhood for a core point in HDBSCAN. Default is None.
    cluster_selection_epsilon (float): A distance threshold. Clusters below this value will be merged. Default is 0.0. higher epslion = fewer, larger clusters
    
    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for encoded Likert responses, UMAP coordinates, and cluster IDs.
    """
    
    # Select columns
    if columns is None and pattern is None:
        raise ValueError("Either 'columns' or 'pattern' must be provided.")
    elif columns is None:
        columns = df.filter(regex=pattern).columns.tolist()
    
    # Encode Likert scales
    df = df.encode_likert(columns, custom_mapping=likert_mapping)
    encoded_columns = [f"likert_encoded_{col}" for col in columns]
    
   
    # Apply UMAP
    df = df.fit_umap(input_columns=encoded_columns, output_columns = ["likert_umap_x", "likert_umap_y"], n_neighbors=umap_n_neighbors, 
                     min_dist=umap_min_dist,
                     metric='cosine')
    
    # Apply HDBSCAN
    df = df.fit_cluster_hdbscan(input_columns=["likert_umap_x", "likert_umap_y"], 
                                output_columns=['question_cluster_id', 'question_cluster_probability'],
                                min_cluster_size=hdbscan_min_cluster_size, 
                                min_samples=hdbscan_min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon
                                )
    
   
    return df


@pf.register_dataframe_method
def encode_likert(df, likert_columns, output_prefix='likert_encoded_', custom_mapping=None, debug=True):
    """
    Encode Likert scale responses to numeric values.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    likert_columns (list): List of column names containing Likert scale responses.
    output_prefix (str): Prefix for the new encoded columns. Default is 'likert_encoded_'.
    custom_mapping (dict): Optional custom mapping for Likert scale responses.
    debug (bool): Prints out the mappings
    
    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for encoded Likert responses.
    """
    
    def default_mapping(response):
        if pd.isna(response):
            return pd.NA
        response = str(response).lower().strip()
        
        # Neutral / Neither / Unsure / Don't know (0)
        if re.search(r'\b(neutral|neither|unsure|know)\b', response) or re.search(r'neither\s+agree\s+nor\s+disagree', response):
            return 0
        
        # Disagree / Dissatisfied (-1)
        if re.search(r'\b(disagree)\b', response) or re.search(r'\b(dis|not|no)[-]{0,1}\s*(agree|satisf)', response):
            return -1
        
        # Agree / Satisfied (1)
        if re.search(r'\bagree\b', response) or re.search(r'satisf', response):
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
        warnings.warn(f"The following phrases were not converted (mapped to NaN): {', '.join(unconverted_phrases)}")
    
    # Alert if default mapping didn't convert everything
    if custom_mapping is None:
        all_responses = set()
        for column in likert_columns:
            all_responses.update(df[column].dropna().unique())
        unconverted = [resp for resp in all_responses if default_mapping(resp) not in [-1, 0, 1]]
        if unconverted:
            warnings.warn(f"The default mapping didn't convert the following responses: {', '.join(unconverted)}")
    
    return df

@pf.register_dataframe_method
def extract_keywords(df: pd.DataFrame, 
                     input_column: str, 
                     output_column: str = 'keywords',
                     preprocessed_column: str = 'preprocessed_text',
                     spacy_column: str = 'spacy_output',
                     lemma_column: str = 'lemmatized_text',
                     top_n: int = 3,
                     threshold: float = 0.4,
                     ngram_range: Tuple[int, int] = (1, 1),
                     min_df: int = 5,
                     min_count: int = None,
                     min_proportion_with_keywords: float = 0.95,
                     **kwargs) -> pd.DataFrame:
    """
    Apply a pipeline of text preprocessing, spaCy processing, lemmatization, and TF-IDF
    to extract keywords from the specified column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to process.
    output_column (str): Name of the column to store the extracted keywords. Default is 'keywords'.
    preprocessed_column (str): Name of the column to store preprocessed text. Default is 'preprocessed_text'.
    spacy_column (str): Name of the column to store spaCy output. Default is 'spacy_output'.
    lemma_column (str): Name of the column to store lemmatized text. Default is 'lemmatized_text'.
    top_n (int): Number of top keywords to extract for each document. Default is 3.
    threshold (float): Minimum TF-IDF score for a keyword to be included. Default is 0.0.
    ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
                         Default is (1, 1) which means only unigrams.
    **kwargs: Additional keyword arguments to pass to the preprocessing, spaCy, lemmatization, or TF-IDF functions.

    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for preprocessed text,
                      spaCy output, lemmatized text, and extracted keywords.
    """

    df_temp = df.copy()
    # Step 1: Preprocess text
    df_temp = df_temp.preprocess_text(input_column=input_column, 
                            output_column=preprocessed_column, 
                            **kwargs.get('preprocess_kwargs', {}))

    df_temp = df_temp.remove_short_comments(input_column=input_column,
                            min_comment_length=5)

    # Step 2: Apply spaCy
    df_temp = df_temp.fit_spacy(input_column=preprocessed_column, 
                      output_column=spacy_column)

    # Step 3: Get lemmatized text
    df_temp = df_temp.get_lemma(input_column=spacy_column, 
                      output_column=lemma_column, 
                      **kwargs.get('lemma_kwargs', {}))

    # Step 4: Apply TF-IDF and extract keywords
    df_temp = df_temp.fit_tfidf(input_column=lemma_column, 
                      output_column=output_column,
                      top_n=top_n,
                      threshold=threshold,
                      ngram_range=ngram_range,
                      min_df=min_df,
                      **kwargs.get('tfidf_kwargs', {}))
    
    df_temp = df_temp.refine_keywords(keyword_column = output_column,
                            text_column = lemma_column,
                            min_proportion = min_proportion_with_keywords,
                            output_column = "refined_keywords",
                            min_count = min_count)

    return df_temp


@pf.register_dataframe_method
def refine_keywords(df: pd.DataFrame, 
                    keyword_column: str = 'keywords', 
                    text_column: str = 'lemmatized_text', 
                    min_count: Union[int, None] = None,
                    min_proportion: float = 0.95,
                    output_column: str = None,
                    debug: bool = True) -> pd.DataFrame:
    """
    Refine keywords by replacing rare keywords with more common ones based on the text content.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    keyword_column (str): Name of the column containing keyword lists.
    text_column (str): Name of the column containing the original text.
    min_count (int, optional): Minimum count for a keyword to be considered common. If None, it will be determined automatically.
    min_proportion (float): Minimum proportion of rows that should have keywords after refinement. Used only if min_count is None. Default is 0.95.
    output_column (str): Column name for the refined keyword output. If it is None, then the keyword_column is over-written.
    debug (bool): If True, print detailed statistics about the refinement process. Default is False.
    
    Returns:
    pd.DataFrame: The input DataFrame with refined keywords.
    """
    if output_column is None:
        output_column = keyword_column

    # Create masked DataFrame
    masked_df, mask = create_masked_df(df, [keyword_column, text_column])
    
    # Step 1 & 2: Collect all keywords and count them
    all_keywords = [keyword for keywords in masked_df[keyword_column] if isinstance(keywords, list) for keyword in keywords]
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
                for common_keyword in sorted(common_keywords, key=lambda k: (-keyword_counts[k], len(k))):
                    if common_keyword in text and common_keyword not in refined_keywords:
                        refined_keywords.append(common_keyword)
                        break
        
        # Ensure correct ordering based on appearance in the original text
        return sorted(refined_keywords, key=lambda k: text.index(k)) if refined_keywords else []

    if min_count is None:
        # Determine min_count automatically
        def get_proportion_with_keywords(count):
            common_keywords = set(keyword_counts[keyword_counts >= count].index)
            refined_keywords = masked_df.apply(lambda row: refine_row_keywords(row, common_keywords), axis=1)
            return (refined_keywords.str.len() > 0).mean()
        
        min_count = 1
        while get_proportion_with_keywords(min_count) > min_proportion:
            min_count += 1
        min_count -= 1  # Go back one step to ensure we're above the min_proportion
    
    # Separate common and rare keywords
    common_keywords = set(keyword_counts[keyword_counts >= min_count].index)
    
    # Apply the refinement to each row
    masked_df[output_column] = masked_df.apply(lambda row: refine_row_keywords(row, common_keywords), axis=1)
    
    # Combine results
    df_to_return = combine_results(df, masked_df, mask, [output_column])
    
    if debug:
        # Calculate statistics
        original_keyword_count = masked_df[keyword_column].apply(lambda x: len(x) if isinstance(x, list) else 0)
        refined_keyword_count = masked_df[output_column].apply(len)
        
        original_unique_keywords = set(keyword for keywords in masked_df[keyword_column] if isinstance(keywords, list) for keyword in keywords)
        refined_unique_keywords = set(keyword for keywords in masked_df[output_column] for keyword in keywords)
        
        print(f"Refinement complete. Min count used: {min_count}")
        print(f"Original average keywords per row: {original_keyword_count.mean():.2f}")
        print(f"Refined average keywords per row: {refined_keyword_count.mean():.2f}")
        print(f"Proportion of rows with keywords after refinement: {(refined_keyword_count > 0).mean():.2%}")
        print(f"Total unique keywords before refinement: {len(original_unique_keywords)}")
        print(f"Total unique keywords after refinement: {len(refined_unique_keywords)}")
        print(f"Reduction in unique keywords: {(1 - len(refined_unique_keywords) / len(original_unique_keywords)):.2%}")
    
    return df_to_return

@pf.register_dataframe_method
def remove_short_comments(df: pd.DataFrame, 
                          input_column: str, 
                          min_comment_length: int = 5) -> pd.DataFrame:
    """
    Replace comments shorter than the specified minimum length with NaN.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    input_column (str): Name of the column containing text to process.
    min_comment_length (int): Minimum length of comment to keep. Default is 5.
    
    Returns:
    pandas.DataFrame: The input DataFrame with short comments replaced by NaN.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Replace short comments with NaN
    df_copy[input_column] = df_copy[input_column].apply(
        lambda x: x if isinstance(x, str) and len(x) >= min_comment_length else np.nan
    )
    
    return df_copy

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
def cluster_comments(df:pd.DataFrame, input_column:str, output_columns:str=["cluster", "cluster_probability"], min_cluster_size=5, cluster_selection_epsilon:float=0.2, n_neighbors:int=15):
    """applies a pipeline of 1) vector embeddings 2) dimensional reduction 3) clustering
    to assign each row a cluster ID so that similar free text comments (found in the input_column) can be grouped together.
    Returns a modified dataframe. If you want control over parameters for the various functions,
    then apply them separately. The defaults should be OK in most cases."""

    df_temp = (df.fit_sentence_transformer(input_column = input_column,
                                                 output_column="sentence_embedding")
                    .fit_umap(input_columns="sentence_embedding",
                              embeddings_in_list=True,
                              n_neighbors=n_neighbors)
                    .fit_cluster_hdbscan(output_columns=output_columns, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon))
    
    return df_temp

@pf.register_dataframe_method
def fit_tfidf(df: pd.DataFrame, 
              input_column: str, 
              output_column: str = 'keywords', 
              top_n: int = 3, 
              threshold: float = 0.6, 
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
    # Inside fit_tfidf function
    tfidf_kwargs['min_df'] = tfidf_kwargs.get('min_df', 1) 

    # Apply TF-IDF vectorization to the masked DataFrame
    tfidf_features, _, feature_names = apply_vectorizer(masked_df, input_column, vectorizer_name='TfidfVectorizer', **tfidf_kwargs)
    
    def extract_top_keywords(row: pd.Series) -> List[str]:
        # Get indices of top N TF-IDF scores
        top_indices = row.nlargest(top_n).index
        
        # Get the original text for this row
        original_text = masked_df.loc[row.name, input_column].lower()
        
        # Filter based on threshold, presence in original text, and get the corresponding feature names
        top_keywords = [
            feature_names[i] for i, idx in enumerate(tfidf_features.columns) 
            if idx in top_indices and row[idx] >= threshold and feature_names[i].lower() in original_text
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
              keep_dep: Union[List[str], None] = ["neg"],
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
    lower_case: bool = False,
    normalize_whitespace: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    flag_short_comments: bool = False,
    min_comment_length: int = 5,
    max_comment_length: int = None,
    remove_punctuation: bool = True,
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
    lower_case (bool): Whether to lowercase all words. Default is False
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
        if lower_case:
            text = text.lower()
        if remove_html:
            text = strip_tags(text)
        
        if normalize_whitespace:
            text = strip_multiple_whitespaces(text)
        
        if remove_numbers:
            text = strip_numeric(text)
        
        if remove_stopwords:
            text = remove_stopwords(text)
        
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