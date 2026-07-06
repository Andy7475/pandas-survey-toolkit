# Changelog

## 2.0.0

### Breaking changes

- **`cluster_questions` now clusters questions, not respondents.** In 1.x it was a
  deprecated alias that clustered *respondents* (one UMAP point per row). It has
  been repurposed to actually cluster **questions** by how respondents co-answer
  them, and it returns a `pandas.Series` (question → cluster id) instead of a
  DataFrame. To cluster respondents, use `cluster_respondents`.
- **Respondent clustering is restructured into three functions:**
  - `cluster_respondents_cosine` — cosine distance + hierarchical (dendrogram)
    clustering (for small/medium surveys). Replaces the 1.x
    `cluster_respondents_correlation`, and now uses **cosine distance instead of
    Pearson correlation**. Cosine compares the *direction* of a respondent's
    answer vector, so people who agree with everything and people who disagree
    with everything separate cleanly; Pearson could not even define a distance
    for those flat "always agree" responses (zero variance).
  - `cluster_respondents_umap` — UMAP + HDBSCAN (for very large surveys). This is
    the 1.x `cluster_respondents`, renamed.
  - `cluster_respondents(method="auto"|"cosine"|"umap")` — a dispatcher that
    picks the engine from the survey size via `_select_clustering_method`.
- **Dropped the `scale` parameter** from `encode_likert` and the clustering
  functions. Encoding is 3-point (−1 / 0 / +1) only; the ±2 / 5-point option was
  removed because cosine distance keys on the *direction* of opinions, not the
  magnitude.
- **Removed the `corr_method` and `distance` (signed/absolute) parameters** — they
  were Pearson-specific and no longer apply to cosine clustering.
- **The heavy free-text NLP libraries are now an optional `nlp` extra.**
  `spacy`, `sentence-transformers`, and `transformers` (and their `torch`
  dependency) moved out of the core dependencies. Install them with
  `pip install "pandas-survey-toolkit[nlp]"`. The core install is torch-free and
  covers Likert clustering and text preprocessing. `gensim` stays core. Calling a
  free-text function (`fit_sentence_transformer`, `extract_sentiment`,
  `fit_spacy`, `extract_keywords`, `cluster_comments`) without the extra raises a
  clear `ImportError` telling you to install it.

### Added

- **`cluster_questions`** — cluster questions from the data (no NLP on the
  question text) via question-vs-question cosine distance + hierarchical
  clustering. Returns a `pd.Series` with the linkage and dendrogram order on
  `.attrs`; save it with `.to_csv(...)` for other analysis.
- **`cluster_survey`** — one call that clusters respondents *and* questions and
  stashes the orderings on `df.attrs`, so `cluster_heatmap_plot` and
  `survey_clustermap` order both axes into readable blocks automatically. The
  respondent method defaults to `"auto"`.
- **`survey_clustermap`** (in `vis`) — a seaborn biclustered clustermap of
  individual respondents × questions with marginal dendrograms, using cosine
  distance and the same red→green sentiment colours as `cluster_heatmap_plot`;
  best for few respondents.

### Changed

- Zero-vector handling: a respondent (or question) that is **all-neutral** has no
  direction, so cosine distance is undefined and it is left unclustered
  (cluster `−1`). Constant *non-neutral* responses (e.g. "agree with everything")
  now cluster normally, which they could not under the old correlation approach.
- `cluster_heatmap_plot` orders the question axis by question cluster (auto-read
  from `df.attrs["question_order"]`) and counts `> 0` / `< 0` for the sentiment
  colours. The colour scheme and cluster-size bar chart are unchanged.
- Clustering helpers store only plain list/dict/scalar objects in `df.attrs`
  (linkages are nested lists, not numpy arrays), so downstream pandas operations
  (`melt`/`concat`/`groupby`) on a clustered frame no longer error.
- `cluster_respondents_cosine` and `cluster_questions` share a single
  cosine-clustering engine (`_cosine_cluster`).

### Developer notes

- CI runs the full matrix with `--extra nlp` **and** a separate torch-free
  `test-core` job (no extra) that asserts `torch`/`spacy` are absent and runs the
  suite; the free-text tests skip themselves via `pytest.importorskip`.
- Run the full test suite locally with `uv sync --group dev --extra nlp`.
