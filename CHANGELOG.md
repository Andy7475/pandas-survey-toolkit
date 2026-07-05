# Changelog

## 2.0.0

### Breaking changes

- **`cluster_questions` now clusters questions, not respondents.** In 1.x it was a
  deprecated alias that clustered *respondents* (one UMAP point per row). It has
  been repurposed to actually cluster **questions** by how respondents co-answer
  them, and it returns a `pandas.Series` (question → cluster id) instead of a
  DataFrame. To cluster respondents, use `cluster_respondents` (UMAP+HDBSCAN) or
  `cluster_respondents_correlation`.
- **The heavy free-text NLP libraries are now an optional `nlp` extra.**
  `spacy`, `sentence-transformers`, and `transformers` (and their `torch`
  dependency) moved out of the core dependencies. Install them with
  `pip install "pandas-survey-toolkit[nlp]"`. The core install is torch-free and
  covers Likert clustering and text preprocessing. `gensim` stays core (it is
  lightweight and used by `preprocess_text` / `clean_survey_columns`). Calling a
  free-text function (`fit_sentence_transformer`, `extract_sentiment`,
  `fit_spacy`, `extract_keywords`, `cluster_comments`) without the extra raises a
  clear `ImportError` telling you to install it.

### Added

- **`cluster_questions`** — cluster questions from the data (no NLP on the
  question text) via question-vs-question correlation + hierarchical clustering.
  Returns a `pd.Series` with the linkage and dendrogram order on `.attrs`; save it
  with `.to_csv(...)` for other analysis.
- **`cluster_survey`** — one call that clusters respondents *and* questions and
  stashes the orderings on `df.attrs`, so `cluster_heatmap_plot` and
  `survey_clustermap` order both axes into readable blocks automatically. The
  respondent method defaults to `"auto"` (correlation for small surveys,
  UMAP+HDBSCAN above `size_threshold`), and warns on an illogical explicit choice.
- **`survey_clustermap`** (in `vis`) — a seaborn biclustered clustermap of
  individual respondents × questions with marginal dendrograms, using the same
  red→green sentiment colours as `cluster_heatmap_plot`; best for few respondents.
- **`encode_likert(scale=5)`** support carried through `cluster_questions` /
  `cluster_survey`.

### Changed

- `cluster_heatmap_plot` now orders the question axis by question cluster
  (auto-read from `df.attrs["question_order"]`, or via a `question_order` arg) and
  counts `> 0` / `< 0` so the sentiment colours are correct on the 5-point (±2)
  encoding. The colour scheme and cluster-size bar chart are unchanged.
- Clustering helpers store only plain list/dict/scalar objects in `df.attrs`
  (linkages are nested lists, not numpy arrays), so downstream pandas operations
  (`melt`/`concat`/`groupby`) on a clustered frame no longer error.
- `cluster_respondents_correlation` and `cluster_questions` share a single
  correlation-clustering engine.

### Developer notes

- CI now runs the full matrix with `--extra nlp` **and** a separate torch-free
  `test-core` job (no extra) that asserts `torch`/`spacy` are absent and runs the
  suite; the free-text tests skip themselves via `pytest.importorskip`.
- Run the full test suite locally with `uv sync --group dev --extra nlp`.
