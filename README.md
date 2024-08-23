# Faster and more Insightful analysis of survey results

This package lets you apply advanced Natural Language Processing (NLP) and Machine Learning functions on survey results directly within a dataframe.

It fills a gap where many NLP packages (like spacy, genism, sentence_transformers) are not designed for data in a spreadsheet (and therefore imported into a dataframe), and  many of the people who are tasked with analysing survey results are often not data scientists.

For example, to extract the sentiment you can just type:

df.extract_sentiment(input_column="survey-comments")

It will abstract away a lot of the data transformation pipeline to give you useful functionality with minimal code.

# Functionality

## Clustering comments
It will group similar free-text comments together and assign a cluster ID. This is a useful step prior to any qualitative analysis.

## Sentiment Analysis
It will measure the sentiment in terms or postive / neutral / negative and assign a score for each of those parts, picking the highest scoring as the most likely overall sentiment.

## Topic analysis
Involves TFIDF and word co-occurence to gain some high level insights into the likely topics

## Clustering likert questions (or other responses)
For strongly disagree ... neutral ... strong agree type responses, it will groups all those questions together to identity groups of respondents within your survey data. This can be much more useful than overall averages across the survey.

## Visualisation
Functions to help make sense of the clusters and topics you have identified using the above functions


