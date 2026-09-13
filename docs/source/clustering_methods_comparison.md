# Clustering a Likert survey: the maths

This note explains how the toolkit clusters Likert-scale survey data on **both
axes** — respondents *and* questions — directly from the response data (no NLP
on the question text), and how to choose between the methods and tune them.

## The short version

- Responses are encoded on a **3-point scale**: −1 (disagree) / 0 (neutral) /
  +1 (agree). Intensity is intentionally ignored.
- Everything is grouped with **cosine distance**, which compares the *direction*
  of a response vector. This is the right choice for opinion data (see §4).
- **Respondents** (`cluster_respondents`): two engines, picked by survey size —
  - `cluster_respondents_cosine`: cosine distance + hierarchical (dendrogram)
    clustering. Best for **small/medium** surveys. `O(n²)` in respondents.
  - `cluster_respondents_umap`: UMAP + HDBSCAN. Best for **very large**
    surveys.
  - `cluster_respondents(method="auto")` picks cosine at/below `size_threshold`
    respondents (default 1000), otherwise umap.
- **Questions** (`cluster_questions`): the same cosine engine applied to the
  columns; returns a `pd.Series` (question → cluster id).
- **Both at once** (`cluster_survey`): clusters both axes and orders the plots.
- **Missing answers** (`nan_strategy`, see §2): `"fill"` (default) treats a gap
  as neutral; `"ignore"` drops the affected respondent.

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

which depends only on the *angle* between the two answer vectors. Those
pairwise distances are the input to agglomerative (average-linkage)
clustering, which produces a dendrogram you cut into clusters. This method is
deterministic. Every respondent is placed into a cluster, except for one case:

- A respondent who answers **everything neutral** is the zero vector — it has
  no direction, so its cosine distance is undefined. Those respondents are not
  assigned to a cluster (cluster **−1**).

### B. UMAP + HDBSCAN (`cluster_respondents_umap`)

UMAP embeds each respondent as one point in 2-D (using the same cosine metric),
then HDBSCAN groups the points by **density**, labelling sparse points as noise
(−1). This method is non-linear and stochastic: the result depends on the
random seed. Because it embeds one point per respondent, it needs many
respondents to estimate a manifold and density.

### C. Questions (`cluster_questions`)

This is the same method as A (with the assumption there will not be > 1000 questions!), applied to columns instead of rows: each
*question* is a vector of the answers it received across respondents, and
questions are clustered by cosine distance between those column vectors.
Returns a `pd.Series` indexed by question name (a question everyone answered
neutral is a zero vector → −1).

---

## 2. Missing answers: `nan_strategy`

Real surveys have unanswered questions. Every function (`cluster_questions`,
`cluster_respondents_cosine`, `cluster_respondents_umap`, and `cluster_respondents`/`cluster_survey` takes `nan_strategy` +
`fill_value`:

- `"fill"` (default): a missing answer is treated as neutral (`fill_value=0`)
  and the respondent/question stays in the clustering.
- `"ignore"`: any respondent with a missing answer among the selected columns
  is dropped from that computation (`-1` for the cosine engines; NaN
  coordinates/cluster id for UMAP).

Both raise a warning when NaNs are found.

Why `nan_strategy` exists: a real all-neutral respondent is already a zero
vector with no direction (§1, unclustered as −1). If every NaN were filled
with zero with no way to choose otherwise, an incomplete respondent would look
the same as a genuinely neutral one.

---

## 3. Choosing between cosine and UMAP: survey size

The two respondent engines perform best at different survey sizes, so
`cluster_respondents` picks between them by size (`_select_clustering_method`):

| Respondents | Method | Why |
|---|---|---|
| ≲ 1000 (`size_threshold`) | **cosine** | exact pairwise distances, a readable dendrogram; `O(n²)` memory is fine here |
| ≫ 1000 (thousands → millions) | **umap** | the `n×n` cosine matrix becomes infeasible; UMAP is ~`O(n log n)`, and density clustering works well when patterns repeat |

If you set `method` explicitly and it does not match the survey size, you get
a warning (cosine is `O(n²)`; UMAP is unreliable with few respondents).

---

## 4. Why cosine (and not correlation or Euclidean)?

Cosine distance fits the structure of opinion data. Pearson correlation and
Euclidean distance do not, for the reasons below.

### vs Pearson correlation

Pearson subtracts each respondent's **mean** before comparing. Two problems
for surveys:

- A respondent who **agrees with everything** (all +1) has zero variance. A
  correlation with them is *undefined*, so they are dropped. But "agrees with
  everything" is a real, common, meaningful segment.
- Centering removes the overall tendency to agree or disagree, which is often
  exactly the signal you want to keep (the difference between broadly positive
  and broadly negative respondents).

Cosine keeps the direction of the raw vector, so all-agree and all-disagree
respondents point in opposite directions (cosine distance 2) and are placed in
separate clusters. The only respondent it cannot place is one who answers
neutral on every question, since that vector has no direction.

> **Worked example.** Four respondents who agree with all 6 questions and four who
> disagree with all 6. Cosine puts the agreers in one cluster and the disagreers
> in another (distance 2 between the groups). Pearson cannot: every respondent has
> zero variance, so no correlation is defined.

### vs Euclidean

Euclidean treats the encoding as a bare number line, so a gap of 2 is a gap of 2
whether or not it crosses neutral. Let's say we have 4 people. One person strongly disagrees (D), one is neutral (N), one just disagrees (d), one agrees (a). The difference between D and N is 2. The difference between d and a is also 2, so we are saying the relative difference between someone who doesn't care (N) and someone who strongly disagrees (D) is the same difference between someone how agrees (a) and disagrees (d), which to me seems wrong.

---

## 5. Choosing the distance threshold

When you cut the dendrogram with `distance_threshold` (instead of `n_clusters`),
two respondents stay in the same cluster while their cosine distance is below the
threshold. **Higher → fewer, larger clusters**; **lower → more, smaller clusters**.

### The rule (exact for ±1 data, no neutral answers)

The cosine distance between two respondents comes from the fraction of
questions they answered oppositely, not the raw count:

```
distance ≈ 2 × (fraction of questions disagreed on)
```

50% disagreement gives a distance of about 1.0, whatever the number of
questions. This is why the same count of disagreements means different things
on different surveys: 5 disagreements is 50% of a 10-question survey
(threshold ≈ 0.9) but only 25% of a 20-question survey (threshold ≈ 0.45).

Written as a raw count `d` out of `Q` questions, the distance is exactly

```
d_cos = 2d / Q
```

To split respondents who disagree on `k` or more questions, set the threshold
halfway between the distance at `k − 1` disagreements and at `k`:

```
threshold ≈ (2k − 1) / Q
```

For Q = 10:

| Disagreements | Fraction | Cosine distance `2d/Q` | Threshold to split at this count `(2d−1)/Q` |
|---|---|---|---|
| 1 | 10% | 0.2 | 0.1 |
| 2 | 20% | 0.4 | 0.3 |
| 3 | 30% | 0.6 | 0.5 |
| 4 | 40% | 0.8 | 0.7 |
| 5 (half) | 50% | 1.0 | 0.9 |

The default `distance_threshold=1.0` is roughly "disagree on half the
questions."

### Effect of neutral answers

The rule above assumes no neutral answers. It does not adjust for them, and
the threshold does not need to be changed to account for them. What actually
happens depends on *which* questions each respondent left neutral:

- **If two respondents are neutral on the same questions**, the distance
  between them reduces exactly to `2d'/Q'`, where `Q'` is the number of
  questions they both answered and `d'` is how many of those they disagreed
  on. It genuinely behaves like a shorter survey containing only the
  questions both of them answered. For example, if both leave the same 10
  questions neutral on a 20-question survey, the comparison behaves like a
  10-question survey.
- **If they are neutral on different questions**, this shortcut does not
  hold. A neutral answer shrinks a respondent's vector toward the origin
  rather than reversing it, so it never flips agreement into disagreement —
  but each respondent's own answers to questions the other left neutral still
  add to their vector's length without adding any shared agreement. This can
  push the distance up even between two respondents who agree on every
  question they both answered. Two respondents who agree on both questions
  they share, but were each neutral on a different set of other questions,
  land at distance 0.33, not 0.

So a neutral answer only shrinks the effective survey size cleanly when
respondents share the same neutral questions. Otherwise, check the actual
distance rather than assuming it from a question count.

### Caveats

- Resolution is `2/Q` per disagreement: coarse with few questions, fine with
  many.
- A larger threshold gives fewer, larger clusters. A smaller threshold gives
  more, smaller clusters. The formula above is a starting point — check the
  actual clusters in the dendrogram (`plot_respondent_dendrogram`) before
  settling on a value.

---

## 6. Clustering questions, and both axes at once (biclustering)

`cluster_questions` clusters the *columns* by cosine distance — a question
everyone agrees with and a question everyone disagrees with point in opposite
directions and are placed in different clusters; a question everyone answers
neutral is a zero vector (−1). It returns a `pd.Series` you can inspect or
`.to_csv()`.

**Biclustering** means clustering both axes and reordering the heatmap so that
coherent blocks appear next to each other. This makes the patterns in a survey
heatmap easier to see. `cluster_survey` does both axes in one call and stores
the orderings so the plotting helpers use them automatically.

---

## 7. How to run

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
cluster_heatmap_plot(survey, respondent_col="respondent_cluster_id",
                     question_cols=[f"likert_encoded_{q}" for q in questions])
survey_clustermap(df, columns=questions, label_col="respondent_id")
```

### One-line summary

Encode ±1, cluster by **cosine distance** (which separates agreers from
disagreers and treats real disagreement as more distant than indifference); use
`cluster_respondents_cosine` for small/medium surveys and `cluster_respondents_umap`
for very large ones (`cluster_respondents` picks automatically); the same engine
clusters questions, and `cluster_survey` does both axes so the heatmaps are
ordered automatically. On a 10-question survey each disagreement is `2/Q ≈ 0.2`
of distance, so a `distance_threshold` near `(2k−1)/Q` splits people who
disagree on `k`+ questions.
