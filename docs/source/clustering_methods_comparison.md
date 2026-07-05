# Clustering a Likert survey: the maths

This note explains how the toolkit clusters Likert-scale survey data on **both
axes** — respondents *and* questions — straight from the response data (no NLP on
the question text), and how to choose between the methods and tune them.

## The short version

- Responses are encoded on a **3-point scale**: −1 (disagree) / 0 (neutral) /
  +1 (agree). Intensity is intentionally ignored.
- Everything is grouped with **cosine distance**, which compares the *direction*
  of a response vector. This is the right choice for opinion data (see §3).
- **Respondents** (`cluster_respondents`): two engines, picked by survey size —
  - `cluster_respondents_cosine`: cosine distance + hierarchical (dendrogram)
    clustering. Best for **small/medium** surveys. `O(n²)` in respondents.
  - `cluster_respondents_umap`: UMAP + HDBSCAN. Scales to **very large** surveys.
  - `cluster_respondents(method="auto")` picks cosine at/below `size_threshold`
    respondents (default 1000), otherwise umap.
- **Questions** (`cluster_questions`): the same cosine engine applied to the
  columns; returns a `pd.Series` (question → cluster id).
- **Both at once** (`cluster_survey`): clusters both axes and orders the plots.

---

## 1. What each method computes

Start from the encoded matrix `X`, shape `(n_respondents, n_questions)`, entries
in {−1, 0, +1}.

### A. Cosine + hierarchical clustering (`cluster_respondents_cosine`)

Each respondent is a row vector of their encoded answers. The distance between
two respondents `a` and `b` is the **cosine distance**

```
d_cos(a, b) = 1 − (a · b) / (‖a‖ ‖b‖)     ∈ [0, 2]
```

which depends only on the *angle* between the two answer vectors. Those pairwise
distances feed agglomerative (average-linkage) clustering, producing a dendrogram
you cut into clusters. Deterministic; every respondent is placed except one
special case:

- A respondent who answers **everything neutral** is the zero vector — it has no
  direction, so its cosine distance is undefined. Those respondents are left
  unclustered (cluster **−1**).

### B. UMAP + HDBSCAN (`cluster_respondents_umap`)

UMAP embeds each respondent as one point in 2-D (using the same cosine metric),
then HDBSCAN groups the points by **density**, labelling sparse points as noise
(−1). Non-linear and stochastic (seed-dependent). Because it embeds one point per
respondent, it needs *many* respondents to estimate a manifold and density.

### C. Questions (`cluster_questions`)

Exactly the mirror of A: each *question* is a vector of the answers it got across
respondents, and questions are clustered by cosine distance between those
column-vectors. Returns a `pd.Series` indexed by question name (a question
everyone answered neutral is a zero vector → −1).

---

## 2. Choosing between cosine and UMAP: survey size

The two respondent engines have opposite sweet spots, so `cluster_respondents`
picks by size (`_select_clustering_method`):

| Respondents | Method | Why |
|---|---|---|
| ≲ 1000 (`size_threshold`) | **cosine** | exact pairwise distances, a readable dendrogram; `O(n²)` memory is fine here |
| ≫ 1000 (thousands → millions) | **umap** | the `n×n` cosine matrix becomes infeasible; UMAP is ~`O(n log n)` and density clustering shines when patterns repeat |

An explicit `method` that is a poor fit for the size warns (cosine is `O(n²)`;
UMAP is unreliable with few respondents).

---

## 3. Why cosine (and not correlation or Euclidean)?

Opinion data has a natural structure that cosine respects and the alternatives do
not.

### vs Pearson correlation

Pearson subtracts each respondent's **mean** before comparing. Two problems for
surveys:

- A respondent who **agrees with everything** (all +1) has zero variance, so a
  correlation with them is *undefined* — they get dropped. But "agrees with
  everything" is a real, common, meaningful segment.
- Centering removes the overall lean, which is often exactly the signal you want
  to keep (the difference between broadly-positive and broadly-negative people).

Cosine keeps the raw direction, so all-agree and all-disagree respondents point
opposite ways (cosine distance 2) and land in separate clusters — the only thing
it cannot place is the genuinely opinion-less all-neutral respondent.

> **Worked example.** Four respondents who agree with all 6 questions and four who
> disagree with all 6. Cosine puts the agreers in one cluster and the disagreers
> in another (distance 2 between the groups). Pearson cannot: every respondent has
> zero variance, so no correlation is defined.

### vs Euclidean

Euclidean treats the encoding as a bare number line, so a gap of 2 is a gap of 2
whether or not it crosses neutral. It rates "one person strongly-agree(+1), the
other neutral(0)" (no real conflict) the same as "one agrees(+1), the other
disagrees(−1)" (a genuine clash). Cosine, working through the dot product, gives
a **negative** contribution to a genuine agree/disagree flip (`+1 × −1 = −1`) but
**zero** to an agree-vs-neutral gap (`+1 × 0 = 0`) — so it correctly treats real
disagreement as more distant than indifference. This is the deeper reason the
toolkit uses cosine on the encoded vectors.

---

## 4. Choosing the distance threshold

When you cut the dendrogram with `distance_threshold` (instead of `n_clusters`),
two respondents stay in the same cluster while their cosine distance is below the
threshold. **Higher → fewer, larger clusters**; **lower → more, smaller clusters**.

### The rule of thumb (exact for ±1 data)

For agree/disagree data encoded ±1 with no neutrals, two respondents are exactly

```
d_cos = 2 × (questions answered oppositely) / (number of questions) = 2 d / Q
```

apart. So **each disagreement moves two respondents `2/Q` further apart** — 0.2
per disagreement on a 10-question survey.

Sensitivity for **Q = 10**:

| Disagreements `d` | Cosine distance `2d/Q` |
|---|---|
| 0 | 0.0 |
| 1 | 0.2 |
| 2 | 0.4 |
| 3 | 0.6 |
| 5 (half) | 1.0 |

### Picking a threshold to split at *k* disagreements

To keep respondents who differ on at most `k − 1` questions together but split
those differing on `k` or more, put the threshold halfway between:

```
threshold ≈ (2k − 1) / Q
```

For **Q = 10**: **0.3** splits at 2+ disagreements, **0.5** at 3+, **0.7** at 4+.
The default `distance_threshold=1.0` corresponds to orthogonal answer directions —
on a 10-question survey, roughly "disagree on 5+ questions".

### Caveats

- **Resolution is `2/Q` per disagreement** — coarse with few questions, fine with
  many.
- **Neutral answers dilute** rather than flip a distance: a neutral shrinks a
  respondent's vector towards the origin instead of pointing it the other way, so
  the clean `2d/Q` count assumes few neutrals.
- **Groups use cophenetic (linkage) distance**, not raw pairwise — use the rule
  for scale, then look at the dendrogram (`plot_respondent_dendrogram`).

---

## 5. Clustering questions, and both axes at once (biclustering)

`cluster_questions` clusters the *columns* by cosine distance — a question
everyone agrees with and a question everyone disagrees with point opposite ways
and separate; a question everyone answers neutral is a zero vector (−1). It
returns a `pd.Series` you can inspect or `.to_csv()`.

**Biclustering** — clustering both axes and reordering the heatmap so coherent
blocks appear — is what makes a survey heatmap explain itself. `cluster_survey`
does both axes in one call and stashes the orderings so the plotting helpers pick
them up automatically.

---

## 6. How to run

```python
import pandas_survey_toolkit.nlp  # registers the DataFrame methods (torch-free)
from pandas_survey_toolkit.vis import (
    cluster_heatmap_plot,
    survey_clustermap,
    plot_respondent_dendrogram,
)

questions = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]

# Respondents: let the dispatcher choose cosine (small) or umap (large)
df_resp = df.cluster_respondents(columns=questions, method="auto")

# ...or call an engine directly:
df_cos = df.cluster_respondents_cosine(
    columns=questions,
    distance_threshold=0.3,   # Q=10: split people who disagree on 2+ questions
)
plot_respondent_dendrogram(df_cos, label_col="respondent_id")

# Questions -> a Series you can save
question_clusters = df.cluster_questions(columns=questions)

# Both axes at once (biclustering) -> feeds the plots
survey = df.cluster_survey(columns=questions)
cluster_heatmap_plot(survey, x="respondent_cluster_id",
                     y=[f"likert_encoded_{q}" for q in questions])
survey_clustermap(df, columns=questions, label_col="respondent_id")
```

### One-line summary

Encode ±1, cluster by **cosine distance** (which separates agreers from
disagreers and treats real disagreement as more distant than indifference); use
`cluster_respondents_cosine` for small/medium surveys and `cluster_respondents_umap`
for very large ones (`cluster_respondents` picks automatically); the same engine
clusters questions, and `cluster_survey` does both axes so the heatmaps order
themselves. On a 10-question survey each disagreement is `2/Q ≈ 0.2` of distance,
so a `distance_threshold` near `(2k−1)/Q` splits people who disagree on `k`+
questions.
