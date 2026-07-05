# Faster and more Insightful analysis of survey results

This package lets you apply advanced Natural Language Processing (NLP) and Machine Learning functions on survey results directly within a dataframe.

It fills a gap where many NLP packages (like spacy, genism, sentence_transformers) are not designed for data in a spreadsheet (and therefore imported into a dataframe), and  many of the people who are tasked with analysing survey results are often not data scientists.

For example, to extract the sentiment you can just type:

df.extract_sentiment(input_column="survey-comments")

It will abstract away a lot of the data transformation pipeline to give you useful functionality with minimal code.

# Installation

```bash
pip install pandas-survey-toolkit           # Likert clustering + text preprocessing (torch-free)
pip install "pandas-survey-toolkit[nlp]"    # + free-text comment tooling (sentence embeddings, spaCy, transformer sentiment)
```

The core install is lightweight and does **not** pull in `torch`/`spacy`, so
clustering Likert questions/respondents works without the heavy deep-learning
stack. Install the `[nlp]` extra when you need free-text comment embeddings,
spaCy pipelines, or transformer sentiment.

# Examples

See [ReadTheDocs](https://pandas-survey-toolkit.readthedocs.io/en/latest/) for simple example notebooks. There are more detailed notebooks in the repo under notebooks/

# Functionality

## Clustering comments
It will group similar free-text comments together and assign a cluster ID. This is a useful step prior to any qualitative analysis.

## Sentiment Analysis
It will measure the sentiment in terms or postive / neutral / negative and assign a score for each of those parts, picking the highest scoring as the most likely overall sentiment.

## Topic analysis
Involves TFIDF and word co-occurence to gain some high level insights into the likely topics

## Clustering a Likert survey (respondents *and* questions)
For strongly disagree ... neutral ... strongly agree responses, the toolkit clusters both axes purely from the data (no NLP on the question text):

Responses are encoded on a simple 3-point scale (−1 / 0 / +1) and grouped with **cosine distance**, which compares the *direction* of people's opinions — so respondents who agree with everything and respondents who disagree with everything separate cleanly.

- **Respondents** — group people who answer along similar lines:
  - `df.cluster_respondents(method="auto")` — picks the engine by survey size (below). This is the one to reach for.
  - `df.cluster_respondents_cosine(...)` — cosine distance + hierarchical (dendrogram) clustering. Best for **small/medium surveys**.
  - `df.cluster_respondents_umap(...)` — UMAP + HDBSCAN. Scales to **very large surveys** (many respondents).
- **Questions** — group questions that get answered similarly:
  - `df.cluster_questions(...)` — returns a `pd.Series` (question → cluster id) you can inspect or save to CSV. (In 2.0 this **clusters questions**; in 1.x it was a deprecated alias that clustered respondents.)
- **Everything at once** — `df.cluster_survey(likert_cols)` clusters respondents and questions and orders both axes so the plots below "just work".

See [`docs/source/clustering_methods_comparison.md`](docs/source/clustering_methods_comparison.md) for the maths, the distance-threshold rule of thumb, and small-survey advice.

## Visualisation
- `cluster_heatmap_plot(...)` — Altair heatmap of sentiment per respondent-cluster × question, with a cluster-size bar chart. Best for **many respondents / few clusters**; questions auto-ordered by their cluster.
- `survey_clustermap(...)` — seaborn biclustered clustermap of individual respondents × questions with marginal dendrograms. Best for **few respondents**, where you can see everyone.
- `plot_respondent_dendrogram(...)` — the respondent dendrogram from cosine clustering.

## Setup
If sentence transformers throws dll errors: https://stackoverflow.com/questions/78484297/c-torch-lib-fbgemm-dll-or-one-of-its-dependencies/78794748#78794748


