# Faster and more Insightful analysis of survey results

This package lets you apply advanced Natural Language Processing (NLP) and machine learning functions on survey results.

It fills a gap where many NLP packages (like spacy, genism, sentence_transformers) are not designed for data in a spreadsheet (and therefore imported into a dataframe).

And that many of the people who are tasked with analysing survey results are often not data scientists.

This package extends pandas with useful functionality like:
df.cluster_comments(input_column="survey-comments", output_column="comment_cluster")

It will abstract away a lot of the data transformation pipeline to give you useful functionality with minimal code:

-> apply sentence transformer -> reduce vector embedding with UMAP -> cluster embeddings with HDBSCAN -> output cluster ID

# Functionality
Core functions include:

## Clustering comments
It will group similar free-text comments together and assign a cluster ID

## Clustering likert questions
For strongly disagree ... neutral ... strong agree type responses, it will groups all those questions together to identity groups of respondents within your survey data.
