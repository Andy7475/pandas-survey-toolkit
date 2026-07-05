# Clustering Likert responses: UMAP+HDBSCAN vs correlation+dendrogram

This note compares the two ways this toolkit can group survey respondents by how
they answer Likert questions, explains the mathematics behind each with small
worked examples, says when each is the right tool (including large surveys), and
gives a rule-of-thumb for choosing the distance threshold.

Both methods answer the *same* question — *"which respondents answer along
similar lines?"* — but they get there through different mathematics and, crucially,
need sample size in **opposite dimensions**.

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

2. **UMAP** builds a fuzzy `k`-nearest-neighbour graph over the respondents using
   that metric, then optimises a 2-D layout by minimising the cross-entropy
   between the high- and low-dimensional fuzzy simplicial sets. It is
   **non-linear, stochastic** (seed-dependent), and preserves *local*
   neighbourhood structure rather than global distances.

3. **HDBSCAN** clusters the 2-D coordinates with a **density** model
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
   r_ij = cov(row_i, row_j) / (σ_i σ_j)
   ```

   Each `r_ij` is estimated **across all the questions**, so more questions means
   each correlation rests on more data.

2. **Correlation → distance.**

   ```
   d_ij = 1 − r_ij         # "signed": opposite responders are far apart
   d_ij = 1 − |r_ij|       # "absolute": group by strength regardless of sign
   ```

3. **Hierarchical (agglomerative) clustering** builds a dendrogram from the
   distance matrix using a linkage rule (average/complete). It is
   **deterministic**, produces a **full hierarchy** you can inspect, has **no
   noise class**, and you cut the tree at a chosen height to get clusters.

Here the *questions* are the observations, so a survey with many questions and
few respondents is the **favourable** case, not the hard one.

### Worked example: the same two respondents, both ways

Five questions, encoded ±1. Respondents A and B agree on four questions and give
opposite answers on one (Q4):

```
A = [+1, +1, +1, −1, −1]
B = [+1, +1, +1, +1, −1]      # differs from A only on Q4
```

- **Cosine distance** = `1 − (A·B)/(‖A‖‖B‖)` = `1 − 3/5` = **0.4**.
- **Pearson distance** = `1 − r` = `1 − 0.612` = **0.39**.

One disagreement out of five ⇒ a distance of ~0.4. Two disagreements out of five
would give ~0.8. That proportionality is the basis of the threshold rule of thumb
in §6.

---

## 2. The core mathematical contrasts

| | UMAP(cosine)+HDBSCAN | Correlation+dendrogram |
|---|---|---|
| Clusters | respondents (rows) | respondents (rows) |
| Needs many… | **respondents** (points to embed) | **questions** (obs per correlation) |
| Scales to large n? | **yes** (~n log n) | **no** (n×n matrix, ~n²) |
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
where it is strongest.

**2. Centering / response style.** Pearson correlation subtracts each
respondent's mean before comparing, so it measures the *shape* of a response
profile, not its overall level. This removes **acquiescence bias** — a tendency
to agree (or rate highly) across the board.

> **Worked example (acquiescence).** Same opinion *shape*, but respondent B rates
> everything a notch higher than A:
> ```
> A = [+2, +1,  0, −1, −2]     mean  0.0
> B = [+2, +2, +1,  0, −1]     mean +0.8   (A shifted up)
> ```
> Pearson distance = `1 − 0.97` = **0.03** (treats them as almost identical — same
> ranking of the questions), while cosine distance = **0.2** (still sees the upward
> shift). For segmentation you usually *want* to ignore that baseline, which is why
> Pearson is the default.

**3. Linear vs non-linear, deterministic vs stochastic.** Correlation captures
linear/monotone association and gives the *same* answer every run; UMAP+HDBSCAN
can recover non-linear structure but varies with the seed and with the density
thresholds. With only three ordinal levels there is little non-linear manifold to
recover, so UMAP's extra flexibility buys little while adding instability — *at
small n*. At large n the trade-off flips (§3).

---

## 3. When UMAP *is* the right tool: large surveys

Everything above argues for correlation+dendrogram on **small** surveys. As the
number of respondents grows, the recommendation flips. Consider **100,000
responses over 10 questions**:

- **Scaling wall.** `cluster_respondents_correlation` builds an
  `n_respondents × n_respondents` distance matrix and links over it — roughly
  `O(n²)` memory and worse in time. At `n = 1,000` that is a 1M-cell matrix
  (fine); at `n = 100,000` it is a 10-billion-cell matrix (~80 GB at float64) and
  the linkage is infeasible. UMAP+HDBSCAN scale far better (UMAP uses approximate
  nearest neighbours, ~`O(n log n)`; HDBSCAN is near-linear), comfortably handling
  10⁵–10⁶ respondents.
- **Density becomes meaningful.** With 10 three-point questions there are at most
  `3¹⁰ ≈ 59,000` possible answer patterns. With 100,000 respondents those
  patterns repeat heavily, so genuine *density* structure emerges — precisely
  what HDBSCAN is built to find, and what a 100,000-leaf dendrogram cannot show.
- **Visualisation.** The 2-D UMAP map lets you *see* the segments and how they
  overlap. That is invaluable at 100k respondents and impossible to read as a
  dendrogram.

So for 100,000 responses over 10 questions, **UMAP+HDBSCAN is the right tool** —
the regime it was designed for.

**Crossover guideline**

| Respondents | Recommended |
|---|---|
| ≲ a few dozen (esp. respondents < questions) | correlation + dendrogram |
| hundreds | either — correlation for interpretability, UMAP for a map |
| thousands → millions | UMAP + HDBSCAN (correlation matrix infeasible; density + visualisation shine) |

One nuance for 10 questions: 10 dimensions is not very high, so you *could* run
HDBSCAN directly on the encoded columns. UMAP still helps by (a) giving a 2-D
picture and (b) smoothing the density estimate. If you skip UMAP, cluster on the
raw encoded columns with a metric that suits the encoding (Euclidean for ±2
intensity, cosine for direction).

---

## 4. "Would UMAP → `df.corr()` on the UMAP coordinates work better?"

**Worse — and not really meaningful.** After UMAP each respondent has just **two**
coordinates (`umap_x`, `umap_y`). Two problems:

- Correlating the two *columns* `umap_x` and `umap_y` gives a single `2×2`
  matrix — how the axes relate across respondents. That is not a
  respondent-by-respondent clustering at all.
- Correlating two *respondents* using only their 2 coordinates means each
  correlation is computed from 2 numbers — statistically empty.

More fundamentally, UMAP is a **lossy, non-distance-preserving, stochastic**
projection: it deliberately preserves only fuzzy *local* topology, not global
linear structure. So correlations on UMAP output do **not** reproduce the
correlations of the original responses.

**Recommendation:** compute correlation on the **original encoded responses**.

There *is* one legitimate hybrid: build the respondent distance matrix from
correlation, **cluster** with hierarchy, and separately feed the *same* matrix to
UMAP (`metric="precomputed"`) purely to get a 2-D **picture**. Clustering stays
trustworthy; UMAP is used only for visualisation.

---

## 5. Encoding sentiment as distance, and adding ±2

The default `encode_likert` (`scale=3`) is three-level: both "agree" and
"strongly agree" map to `+1`. Passing `scale=5` preserves **intensity**:

```
strongly disagree = −2, disagree = −1, neutral = 0, agree = +1, strongly agree = +2
```

Why intensity changes the geometry:

- **Under Euclidean / correlation distance**, ±2 makes the gap between "strongly
  disagree" and "strongly agree" (4) twice the mild gap (2), so distances reflect
  *how far apart* two sentiments are.
- **Under cosine, ±2 is partly wasted** — cosine is invariant to each
  respondent's magnitude.

> **Worked example (uniform intensity is invisible).** If one respondent is simply
> a scaled-up copy of another —
> ```
> A = [+2, +2, −2, +2, −2]      (strong opinions)
> B = [+1, +1, −1, +1, −1]      (same pattern, mild)   = A / 2
> ```
> — then **both** cosine and Pearson give distance **0**. Uniform intensity does not
> change direction or shape. ±2 only matters when intensity *varies across
> questions* (someone strong on some items, mild on others). Keep that in mind
> before reaching for `scale=5`: it helps when the *pattern* of intensity carries
> signal, not when one person is simply more emphatic overall.

Two caveats on the ordinal nature of Likert data:

- **Equal spacing.** Encoding `{−2,−1,0,1,2}` treats strongly→mild as equal to
  mild→neutral — the standard *interval* treatment; defensible but an assumption.
- **Pearson vs Spearman vs polychoric.** Spearman (rank) needs only monotonicity,
  but *between respondents* it ranks the questions within each respondent — a
  slightly different construct, so **Pearson is the more natural default** for
  respondent-vs-respondent similarity (`corr_method` lets you switch). Polychoric
  correlation — modelling a latent continuous variable behind the ordinal
  responses — is the gold standard if you outgrow Pearson/Spearman.

---

## 6. Choosing the distance threshold

When you cut the dendrogram with `distance_threshold` (rather than `n_clusters`),
two respondents stay in the same cluster while their distance is below the
threshold. **Higher threshold → fewer, larger clusters** (more disagreement
tolerated before people are split); **lower threshold → more, smaller clusters**
(a single differing answer can separate people). At the extremes: a threshold
above the tree's tallest merge gives one big cluster; a threshold near 0 makes
every distinct response pattern its own cluster.

### The rule of thumb (±1 encoding)

For agree/disagree data encoded as ±1, two respondents sit about

```
distance ≈ 2 × (number of questions answered oppositely) / (number of questions)
         =  2 d / Q
```

apart. This is **exact for cosine** and **very close for Pearson** (the default,
which runs a touch lower because it re-centres). So **each disagreement moves two
respondents ~`2/Q` further apart** — 0.2 per disagreement for a 10-question
survey.

Sensitivity table for **Q = 10** (computed, ±1 encoding):

| Disagreements `d` | Cosine distance (`2d/Q`) | Pearson distance (default) |
|---|---|---|
| 0 | 0.0 | 0.00 |
| 1 | 0.2 | 0.18 |
| 2 | 0.4 | 0.34 |
| 3 | 0.6 | 0.50 |
| 4 | 0.8 | 0.67 |
| 5 (half) | 1.0 | ~1.0 (r ≈ 0) |

So for your example — *two people who answer the same on 9 questions and opposite
on 1* — the distance between them is about **0.2**. A threshold **above 0.2** keeps
them together; a threshold **below 0.2** splits them.

### Picking a threshold to split at *k* disagreements

To keep respondents who differ on at most `k − 1` questions together but split
those differing on `k` or more, put the threshold halfway between the `k−1` and
`k` rows:

```
threshold ≈ (2k − 1) / Q
```

For **Q = 10**:

| Split when respondents disagree on… | Threshold ≈ |
|---|---|
| 2 or more questions | **0.3** |
| 3 or more questions | **0.5** |
| 4 or more questions | **0.7** |
| 5 or more (the default `1.0`) | **1.0** |

Because the default Pearson distances sit a little below `2d/Q`, lean to the low
side of each band if you must *guarantee* the split (e.g. 0.25 rather than 0.30 to
be sure of separating 2+ disagreers).

The built-in default `distance_threshold=1.0` corresponds to `r > 0` — it only
separates respondents who disagree on **more than half** the questions (5+ of 10),
i.e. broadly-opposed groups.

### Caveats

- **Resolution is `2/Q` per disagreement.** With few questions each disagreement
  is a big jump (coarse control); with many questions you get finer control. On a
  4-question survey each disagreement is worth 0.5, so you can only split at
  "1+", "2+", etc. — nothing in between.
- **Neutrals and ±2 change the per-question weight.** A neutral-vs-agree question
  is a half step; a strongly-agree-vs-strongly-disagree on the ±2 scale counts
  double. The "count of disagreements" reading is cleanest for pure ±1
  agree/disagree data.
- **Groups use cophenetic, not pairwise, distance.** The `2d/Q` formula is exact
  for a *pair*. For groups the cut uses the linkage tree's merge heights, so with
  average/complete linkage a cluster only holds when its members are *mutually*
  close — effective splitting is a little stricter than the pairwise rule
  suggests. Use the rule for scale, then look at the dendrogram
  (`plot_respondent_dendrogram`) to place the cut.

---

## 6b. Note: when does `distance="absolute"` make sense?

The default, `distance="signed"` (`1 − r`), puts respondents who answer
*oppositely* as far apart as possible — two perfect opposites (`r = −1`) sit at
distance 2. `distance="absolute"` (`1 − |r|`) instead puts perfect opposites at
distance **0**, i.e. it groups respondents by *how strongly their answers are
related*, ignoring whether they agree or disagree.

For ordinary "segment people by their opinions", that is usually **wrong** — you
do not want to merge the pro and anti camps. So default to signed. But absolute
is genuinely the right choice in a few situations:

1. **Two-stage polarisation analysis — "who is in the debate?" before "which
   side?"** Absolute distance groups everyone aligned to the dominant axis of
   (dis)agreement — *both* camps — into one bloc, and separates out respondents
   whose answers do not track that axis. Worked example (Q = 8): a pro pair
   `P1, P2`, an anti pair `A1, A2` (mirror images of the pro pair), and one
   "off-axis" respondent `X` whose answers are uncorrelated with the main split.

   | 2 clusters | Result |
   |---|---|
   | `distance="signed"` | `{P1,P2}` vs `{A1,A2}` — splits by side |
   | `distance="absolute"` | `{P1,P2,A1,A2}` vs `{X}` — the polarised bloc vs the off-axis respondent |

   So a natural workflow is: run **absolute** first to find the bloc engaged with
   the dominant fault line, then run **signed** within that bloc to split it into
   the two opposing camps.

2. **Mirror pairs / mutual predictability.** `|r| = 1` means one respondent's
   answers perfectly predict the other's — whether identical or flipped. If a
   perfect opposite is a *strong* relationship by design (adversarial or
   complementary dyads: buyer/seller, prosecution/defence, matched opponents),
   absolute distance treats "moves in lockstep" and "moves in exact opposition"
   as equally close, which is what you want. (In the example above, cutting to 3
   clusters with absolute distance pairs each respondent with its exact mirror:
   `{P1,A1}`, `{P2,A2}`, `{X}`.)

3. **Robustness to unresolved reverse-keyed items.** If some items are
   reverse-worded and have *not* been reverse-scored, respondents with the same
   underlying attitude can look anti-correlated purely from the sign convention.
   `|r|` makes the clustering invariant to that flip. (Cleaner still is to
   reverse-score the items, but absolute is a useful safety net.)

In short: **signed** to find *opinion camps*, **absolute** to find the *axis of
disagreement itself* (and who lies on it, either pole).

---

## 7. Review and recommendations for surveys with few respondents

1. **Prefer `cluster_respondents_correlation` when respondents < questions.** It
   is statistically strong because each correlation is estimated over many
   questions, and it always returns a hierarchy you can cut, whereas UMAP+HDBSCAN
   needs many points and, with default `min_cluster_size`, otherwise errors or
   labels everyone noise.
2. **Watch the degenerate rows.** An **all-neutral** respondent is the zero
   vector (cosine undefined); a respondent who gives the **same answer to every
   question** has zero variance (correlation undefined).
   `cluster_respondents_correlation` detects the latter, warns, and leaves those
   respondents unclustered (`−1`).
3. **Strip acquiescence bias** by using Pearson (default) or row-standardising
   each respondent, so grouping is by response *shape*.
4. **Right-size the density parameters** for the UMAP path — the default
   `hdbscan_min_cluster_size=20` is far too large for a small survey;
   `cluster_respondents` warns and shrinks it, but the correlation method avoids
   the problem entirely.
5. **Choose the distance to match intent** — `distance="signed"` (`1 − r`) puts
   opposite responders far apart; `distance="absolute"` (`1 − |r|`) groups people
   whose opinions move together *or* in exact opposition.
6. **Use `scale=5` only if intensity varies across questions** (see §5), and pair
   it with a magnitude-sensitive metric (Euclidean or correlation), not cosine.

---

## 8. How to run each approach

```python
import pandas_survey_toolkit.nlp  # registers the DataFrame methods
from pandas_survey_toolkit.vis import plot_respondent_dendrogram

questions = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]

# Many respondents (thousands+): density-based segmentation and a 2-D map
df_umap = df.cluster_respondents(columns=questions)          # respondent_cluster_id

# Few respondents / many questions: correlation + dendrogram
df_corr = df.cluster_respondents_correlation(
    columns=questions,
    corr_method="pearson",
    distance="signed",          # 1 - r
    distance_threshold=0.3,     # Q=10: split people who disagree on 2+ questions
)
plot_respondent_dendrogram(df_corr, label_col="respondent_id")
```

### One-line summary

UMAP+HDBSCAN and correlation+dendrogram both cluster *respondents*, but UMAP needs
many respondents (and is the right tool at tens of thousands) while correlation
needs many questions (and wins on small surveys); on a 10-question survey each
disagreement is worth about `2/Q ≈ 0.2` of distance, so a `distance_threshold`
near `(2k−1)/Q` splits people who disagree on `k` or more questions.
