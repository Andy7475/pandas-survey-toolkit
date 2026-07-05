# Faster and more Insightful analysis of survey results

This package lets you apply advanced Natural Language Processing (NLP) and Machine Learning functions on survey results directly within a dataframe.

It fills a gap where many NLP packages (like spacy, genism, sentence_transformers) are not designed for data in a spreadsheet (and therefore imported into a dataframe), and  many of the people who are tasked with analysing survey results are often not data scientists.

For example, to extract the sentiment you can just type:

df.extract_sentiment(input_column="survey-comments")

It will abstract away a lot of the data transformation pipeline to give you useful functionality with minimal code.

# Examples

See [ReadTheDocs](https://pandas-survey-toolkit.readthedocs.io/en/latest/) for simple example notebooks. There are more detailed notebooks in the repo under notebooks/

# Functionality

## Clustering comments
It will group similar free-text comments together and assign a cluster ID. This is a useful step prior to any qualitative analysis.

## Sentiment Analysis
It will measure the sentiment in terms or postive / neutral / negative and assign a score for each of those parts, picking the highest scoring as the most likely overall sentiment.

## Topic analysis
Involves TFIDF and word co-occurence to gain some high level insights into the likely topics

## Clustering respondents by their Likert answers
For strongly disagree ... neutral ... strongly agree type responses, this groups respondents who answer along similar lines, which can be far more useful than overall averages across the survey.

Two approaches are available:

- `df.cluster_respondents(...)` — UMAP (cosine) + HDBSCAN. Best when you have **many respondents**.
- `df.cluster_respondents_correlation(...)` — respondent correlation + hierarchical (dendrogram) clustering. Best when you have **few respondents relative to the number of questions**, because each correlation is estimated across all the questions.

`encode_likert(..., scale=5)` keeps the intensity of agreement (strongly agree/disagree → ±2). See [`docs/source/clustering_methods_comparison.md`](docs/source/clustering_methods_comparison.md) for the maths behind the two methods and guidance on encoding and small surveys. (`cluster_questions` is a deprecated alias for `cluster_respondents`.)

## Visualisation
Functions to help make sense of the clusters and topics you have identified using the above functions (in development)

## Setup
If sentence transformers throws dll errors: https://stackoverflow.com/questions/78484297/c-torch-lib-fbgemm-dll-or-one-of-its-dependencies/78794748#78794748


