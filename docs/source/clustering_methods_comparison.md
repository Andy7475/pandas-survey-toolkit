# Clustering Likert responses: UMAP+HDBSCAN vs correlation+dendrogram

This note compares the two ways this toolkit can group survey respondents by how
they answer Likert questions, explains the mathematics behind each, and gives
guidance for the common awkward case: **few respondents relative to the number of
questions.**

Both methods answer the *same* question — *"which respondents answer along
similar lines?"* — but they get there through very different mathematics, and,
crucially, they need sample size in **opposite dimensions**.

- `cluster_respondents` — UMAP (cosine) → HDBSCAN. Needs **many respondents**.
- `cluster_respondents_correlation` — respondent correlation → hierarchical
  (dendrogram) clustering. Needs **many questions**, and is the better choice
  when respondents are scarce.

> Note on naming: the original method was called `cluster_questions`, but it has
> always clustered *respondents* (one embedded point per row), not questions. It
> is now `cluster_respondents`; `cluster_questions` remains as a deprecated
> alias.

---

## 1. What each method actually computes

Start from the encoded response matrix `X` with shape `(n_respondents,
n_questions)`, where each entry is the encoded Likert value (e.g. −1/0/+1).

### A. UMAP (cosine) → HDBSCAN

1. **Cosine distance between respondents.** For two respondent vectors `a` and
   `b` (each of length `n_questions`):

   ```
   d_cos(a, b) = 1 − (a · b) / (‖a‖ ‖b‖)
   ```

   This depends only on the *angle* between the vectors — it is invariant to
   each respondent's overall magnitude.

2. **UMAP** builds a fuzzy `k`-nearest-neighbour graph over the respondents
   using that metric, then optimises a low-dimensional (2-D) layout by
   minimising the cross-entropy between the high-dimensional and low-dimensional
   fuzzy simplicial sets. It is **non-linear, stochastic** (seed-dependent), and
   preserves *local* neighbourhood structure rather than global distances.

3. **HDBSCAN** then clusters the 2-D coordinates using a **density** model
   (mutual-reachability distances → minimum spanning tree → condensed tree →
   cluster extraction by stability). Points in no dense region are labelled
   **noise** (`−1`).

Each respondent is **one point** for UMAP to place. UMAP needs enough points to
estimate a manifold, and HDBSCAN needs enough points to estimate density.

### B. Respondent correlation → dendrogram

1. **Correlation between respondents.** Transpose so respondents are the
   variables and questions are the observations, then correlate:

   ```
   R = X.T.corr()          # shape (n_respondents, n_respondents)
   r_ij = cov(Q·i, Q·j) / (σ_i σ_j)
   ```

   Each `r_ij` is estimated **across all the questions**. So *more questions*
   means each correlation is estimated from more data.

2. **Correlation → distance.** Convert similarity to a distance, e.g.

   ```
   d_ij = 1 − r_ij         # "signed": opposite responders are far apart
   d_ij = 1 − |r_ij|       # "absolute": group by strength regardless of sign
   ```

   (`√(2(1 − r))` is the proper metric form and is monotonic in `1 − r`, so it
   gives the same tree for the linkages we use.)

3. **Hierarchical (agglomerative) clustering** builds a dendrogram from the
   distance matrix using a linkage rule (average/complete). It is
   **deterministic**, produces a **full hierarchy** you can inspect, has **no
   noise class** (everyone is placed), and cutting the tree at a chosen height
   yields the clusters.

Here the *questions* are the observations, so a survey with many questions and
few respondents is the **favourable** case, not the hard one.

---

## 2. The core mathematical contrasts

| | UMAP(cosine)+HDBSCAN | Correlation+dendrogram |
|---|---|---|
| Clusters | respondents (rows) | respondents (rows) |
| Needs many… | **respondents** (points to embed) | **questions** (obs per correlation) |
| Centering | none (raw direction) | per-respondent mean removed |
| Model | non-linear manifold + density | linear/monotone association + hierarchy |
| Determinism | stochastic (seed-dependent) | deterministic |
| Unassigned points | possible (noise = −1) | none (all placed) |
| Small-n behaviour | breaks down | degrades gracefully |

Three of these deserve emphasis.

**1. Opposite sample-size requirements (the key point).** UMAP places one point
per respondent and HDBSCAN needs dense regions of those points; both fail when
respondents are few. Respondent-correlation instead estimates each pairwise
similarity across the questions, so few-respondents/many-questions is exactly
where it is strongest. This single fact explains why the correlation route wins
on small surveys.

**2. Centering / response style.** Pearson correlation subtracts each
respondent's mean before comparing, so it measures the *shape* of a response
profile, not its overall level. This removes **acquiescence bias** (a respondent
who agrees with everything, just less strongly) — two "positive" respondents are
judged similar even if one is consistently more enthusiastic. Cosine keeps raw
magnitude/direction and does **not** remove that baseline. (Formally,
Pearson-between-respondents equals the cosine of the *mean-centred* respondent
vectors.) For segmentation, removing acquiescence is usually what you want.

**3. Linear vs non-linear, deterministic vs stochastic.** Correlation captures
linear/monotone association and gives the *same* answer every run; UMAP+HDBSCAN
can recover non-linear structure but varies with the random seed and with the
density thresholds (`min_cluster_size`, `cluster_selection_epsilon`). With only
three ordinal levels there is little non-linear manifold to recover, so UMAP's
extra flexibility buys little while adding instability.

---

## 3. "Would UMAP → `df.corr()` on the UMAP coordinates work better?"

**Worse — and not really meaningful.** After UMAP each respondent has just **two**
coordinates (`umap_x`, `umap_y`). Two problems:

- Correlating the two *columns* `umap_x` and `umap_y` gives a single `2×2`
  matrix — one number describing how the axes relate across respondents. That is
  not a respondent-by-respondent clustering at all.
- Correlating two *respondents* using only their 2 coordinates means each
  correlation is computed from 2 numbers — statistically empty.

More fundamentally, UMAP is a **lossy, non-distance-preserving, stochastic**
projection: it deliberately preserves only fuzzy *local* topology, not global
linear structure. So correlations computed on UMAP output do **not** reproduce
the correlations of the original responses. Feeding UMAP coordinates into
`df.corr()` compounds UMAP's distortion instead of measuring response similarity.

**Recommendation:** compute correlation on the **original encoded responses**, not
on UMAP coordinates.

There *is* one legitimate hybrid worth knowing: build the respondent
distance matrix from correlation, **cluster** with hierarchy, and separately feed
the *same* distance matrix to UMAP (`metric="precomputed"`) purely to get a 2-D
**picture**. That keeps the clustering trustworthy (correlation + dendrogram) and
uses UMAP only for visualisation — the two concerns are decoupled.

---

## 4. Encoding sentiment as distance, and adding ±2

The default `encode_likert` (`scale=3`) is strictly three-level: both "agree" and
"strongly agree" map to `+1`. Passing `scale=5` preserves **intensity**:

```
strongly disagree = −2, disagree = −1, neutral = 0, agree = +1, strongly agree = +2
```

Why intensity changes the geometry:

- **Under Euclidean / correlation distance**, ±2 makes the gap between "strongly
  disagree" and "strongly agree" (4) twice the mild gap (2). Distances now
  reflect *how far apart* two sentiments are, which is usually what you want when
  clustering by opinion.
- **Under cosine, ±2 is partly wasted.** Cosine is invariant to each respondent's
  magnitude, so a mildly-positive and a strongly-positive respondent can point in
  nearly the same direction and be judged similar regardless of intensity. **If
  sentiment intensity should count towards distance, do not use cosine** — prefer
  Euclidean, or a centred measure like Pearson correlation.

Two caveats on the ordinal nature of Likert data:

- **Equal spacing.** Encoding `{−2,−1,0,1,2}` treats the gap strongly→mild as
  equal to mild→neutral. This is the standard *interval* treatment of Likert
  data; defensible and widely used, but an assumption.
- **Pearson vs Spearman vs polychoric.** Spearman (rank) correlation is the usual
  ordinal-safe choice and needs only monotonicity, not equal spacing — but note
  that Spearman *between respondents* ranks the questions within each respondent,
  a slightly different construct. For respondent-vs-respondent similarity,
  **Pearson is the more natural default** (`cluster_respondents_correlation` uses
  it, and exposes `corr_method` if you want to switch). Polychoric correlation —
  which models a latent continuous variable behind the ordinal responses — is the
  gold standard for Likert items and is the natural next step if you outgrow
  Pearson/Spearman.

---

## 5. Review and recommendations for surveys with few respondents

1. **Prefer `cluster_respondents_correlation` when respondents < questions.** It
   is statistically strong precisely because each correlation is estimated over
   many questions, and it degrades gracefully — it always returns a hierarchy you
   can cut, whereas UMAP+HDBSCAN needs many points and, with default
   `min_cluster_size`, will otherwise error or label everyone noise.
2. **Watch the degenerate rows.** An **all-neutral** respondent is the zero
   vector, so cosine is undefined; a respondent who gives the **same answer to
   every question** has zero variance, so their correlation is undefined.
   `cluster_respondents_correlation` detects the latter, warns, and leaves those
   respondents unclustered (`−1`).
3. **Strip acquiescence bias** by using Pearson (default) or by row-standardising
   each respondent, so grouping is by response *shape*, not overall positivity.
4. **Right-size the density parameters** for the UMAP path. The default
   `hdbscan_min_cluster_size=20` is far too large for a small survey;
   `cluster_respondents` now warns and shrinks it, but the correlation method
   avoids the problem entirely.
5. **Choose the distance to match intent.** Use `distance="signed"` (`1 − r`) to
   put opposite responders far apart, or `distance="absolute"` (`1 − |r|`) to
   group respondents whose opinions move together *or* in exact opposition.
6. **Use `scale=5` if intensity matters**, and pair it with a magnitude-sensitive
   metric (Euclidean or correlation), not cosine.

---

## 6. How to run each approach

```python
import pandas_survey_toolkit.nlp  # registers the DataFrame methods
from pandas_survey_toolkit.vis import plot_respondent_dendrogram

questions = ["q1", "q2", "q3", "q4", "q5", "q6"]

# Many respondents: density-based segmentation
df_umap = df.cluster_respondents(columns=questions)          # respondent_cluster_id

# Few respondents / many questions: correlation + dendrogram
df_corr = df.cluster_respondents_correlation(
    columns=questions,
    scale=5,              # keep agreement intensity
    corr_method="pearson",
    distance="signed",    # 1 - r
    n_clusters=3,         # or distance_threshold=...
)
plot_respondent_dendrogram(df_corr, label_col="respondent_id")
```

### One-line summary

UMAP+HDBSCAN and correlation+dendrogram both cluster *respondents*, but UMAP needs
many respondents while correlation needs many questions — so on a small survey,
correlate the respondents across the questions and cut a dendrogram; reserve
UMAP+HDBSCAN for when you have plenty of respondents, and if you keep intensity in
the encoding (±2), pair it with a magnitude-aware metric rather than cosine.
