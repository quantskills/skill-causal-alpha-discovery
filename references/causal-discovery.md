# Causal Discovery for OHLCV Alpha Factors

This reference describes the causal discovery methodology that elevates alpha
factor mining from pure correlation to **causal inference**. This is the key
differentiator: causal factors are stable across market regimes, while
correlational factors degrade when the data-generating process changes.

---

## 1. Motivation: Why Causality Matters for Alpha

### The Problem with Correlation Mining

Traditional alpha factor mining answers the question:

> "When this OHLCV pattern appeared in the past, what happened to returns?"

This is **correlational** reasoning. The problem: financial markets are
non-stationary. The correlation between any pattern and future returns can
(and does) change across regimes.

**Example**: A typical correlational factor might show:

| Period | Best Factor Sharpe |
|--------|-------------------|
| 2015–2019 (train) | ~1.4 |
| 2020–2025 (test) | ~0.7 |

That's a **50% Sharpe decay** out-of-sample. This is classic correlation
breakdown.

### The Causal Alternative

Causal discovery answers a fundamentally different question:

> "Does this OHLCV-derived quantity *cause* future returns through a stable
> structural mechanism?"

If the answer is yes, the relationship is **invariant** — it holds regardless
of bull/bear/sideways/volatile regimes. Causal factors are discovered, not
fitted.

---

## 2. Causal Discovery Algorithms

The skill implements three complementary algorithms:

### 2.1 PC Algorithm (Constraint-Based)

**What it does**: Tests conditional independencies to discover the causal
skeleton (undirected graph of which variables are related).

**How it works**:
1. Start with a fully connected undirected graph over all variables
2. For each pair (X, Y), test X ⊥ Y | S for conditioning sets S of
   increasing size
3. Remove edge (X, Y) if any conditional independence is found
4. Orient edges using collider detection (V-structures)

**Key parameter**: `pc_alpha` — significance level for independence tests.
Lower = sparser graph (fewer edges, more conservative).

**Strengths**: Non-parametric (doesn't assume linearity), well-understood
statistical properties.

**Weaknesses**: Exponential complexity in number of variables. Practical
limit ~30 variables.

### 2.2 DirectLiNGAM (Functional Causal Discovery)

**What it does**: Discovers causal ordering AND estimates effect magnitudes
under the assumption of linear non-Gaussian additive noise.

**How it works**:
1. Identify the most exogenous variable (residuals most independent of others)
2. Remove it, regress remaining variables on it, iterate
3. Build lower-triangular adjacency matrix from regression coefficients
4. Prune weak edges

**Key assumption**: The data follows $X_i = \sum b_{ij} X_j + e_i$ where
$e_i$ are non-Gaussian and mutually independent. Financial data often
satisfies this (returns are heavy-tailed, non-Gaussian).

**Strengths**: Fully orients edges (not just skeleton), estimates effect
magnitudes, computationally efficient.

**Weaknesses**: Assumes linearity and no latent confounders.

### 2.3 NOTEARS (Continuous Optimization)

**What it does**: Reformulates DAG learning as a continuous optimization
problem with a differentiable acyclicity constraint.

**DAG constraint**:
$$h(W) = \text{tr}(e^{W \odot W}) - d = 0 \iff W \text{ encodes a DAG}$$

This means we can use **gradient descent** to learn the DAG structure,
scaling to hundreds of variables.

**Strengths**: Scales to high dimensions, GPU-compatible.

**Weaknesses**: Linear SEM assumption, requires careful hyperparameter tuning.

### 2.4 Rank-IC Selection (Default, Recommended)

**What it does**: Computes cross-sectional rank IC between each feature and
the target, then selects top features by absolute IC. Applies diversity
constraints to avoid near-duplicate features.

**How it works**:
1. For each feature, compute daily rank IC against target
2. Average IC across all dates → feature score
3. Group features by base name (stripping trailing `_\d+d` lookback suffix)
4. Select up to `max_components` features, max `diversity_max_per_family` per base group
5. Also excludes any `forward_return_*` feature to prevent look-ahead bias

**Strengths**: Extremely fast (handles 300K+ observations × 88+ features),
interpretable, avoids redundant signals. No hyperparameter tuning needed.

**Config**: `causal_discovery.method: "ic_selection"`,
`causal_discovery.diversity_max_per_family: 1` (default).

### 2.5 Hybrid Approach (Alternative)

For deeper analysis on smaller datasets, combine:
1. **PC algorithm** finds the skeleton — robust, non-parametric
2. **LiNGAM** orients edges and estimates effect sizes
3. **Bootstrap** (100 runs) computes edge stability scores

---

## 3. Structural Causal Model (SCM) & Do-Calculus

Once the DAG is discovered, we build a Structural Causal Model:

$$X_i := f_i(PA_i, U_i), \quad U_i \perp U_j$$

Where $PA_i$ are the causal parents of $X_i$ and $U_i$ are independent
exogenous noise terms.

### Three Levels of Causal Reasoning

**Level 1 — Association** (standard ML):
$$P(\text{return} \mid \text{vol\_20d} = 0.03)$$
"What's the expected return when 20-day vol is 3%?"

**Level 2 — Intervention** (do-calculus):
$$P(\text{return} \mid do(\text{vol\_20d} = 0.03))$$
"What would happen to returns if we *forced* 20-day vol to be 3%?"

**Level 3 — Counterfactual** (retrospective):
"Given that vol was 3% and the stock returned +2%, what would the return
have been if vol had been 1% instead?"

### Backdoor Adjustment

The ATE (Average Treatment Effect) of a causal parent $P$ on returns $R$ is:

$$ATE = \mathbb{E}[R \mid do(P = p + \delta)] - \mathbb{E}[R \mid do(P = p)]$$

By the backdoor criterion, if $Z$ blocks all backdoor paths from $P$ to $R$:

$$P(R \mid do(P = p)) = \sum_z P(R \mid P = p, Z = z) \cdot P(Z = z)$$

The skill identifies backdoor adjustment sets from the causal DAG
automatically.

---

## 4. Regime Invariance: The Key Test

The **defining property of a causal relationship** is invariance under
environmental changes (Peters et al., 2016; Arjovsky et al., 2019).

### Invariance Hypothesis

> $H_0$: The ATE of feature $X$ on forward returns is equal across all
> market regimes.
>
> $H_1$: The ATE varies by regime (indicating a spurious correlation).

### Test Procedure

1. **Identify regimes**: Split data into 5–7 distinct market regimes using
   volatility clustering, return quantiles, or calendar years.

2. **Estimate ATE per regime**: Run linear regression of returns on factor
   within each regime. Bootstrap standard errors.

3. **χ² homogeneity test**: Test whether ATE estimates differ across regimes:
   $$\chi^2 = \sum_{r=1}^R \frac{(ATE_r - ATE_{pooled})^2}{SE_r^2} \sim \chi^2(R-1)$$

4. **Sign consistency**: Check that the direction of the effect doesn't flip
   across regimes.

5. **Certificate**: Produce a formal invariance certificate with confidence
   level.

### Interpretation

| p-value | Sign Consistency | Verdict |
|---------|-----------------|---------|
| > 0.05 | > 90% | **Strongly Invariant** — genuine causal mechanism |
| > 0.05 | < 90% | **Moderately Invariant** — mostly stable |
| < 0.01 | any | **Regime Dependent** — spurious correlation |
| 0.01–0.05 | any | **Weakly Regime Dependent** — proceed with caution |

---

## 5. From Causal Graph to Tradable Alpha

The skill converts causal discoveries into valid OHLCV factor expressions:

### Step 1: Identify Causal Parents

From the DAG, extract all variables that are **direct causal parents** of
`forward_return_5d` (or whichever target horizon you choose).

### Step 2: Filter by Stability

Keep only edges that appear in ≥70% of bootstrap runs. This eliminates
spurious edges that are artifacts of the specific sample.

### Step 3: Map to OHLCV Expressions

Each causal parent feature is mapped back to a valid expression using only
the 6 OHLCV fields and 27 allowed functions. The mapping is defined in
`scripts/build_causal_alpha.py` (`FEATURE_TO_EXPRESSION` dict).

### Step 4: Weight by ATE × Stability

Each component is weighted by the product of its Average Treatment Effect
and its bootstrap stability:

$$w_i = |ATE_i| \times stability_i$$

Normalized weights determine each component's contribution to the final
factor.

### Step 5: Combine into Single Expression

The final alpha is a weighted sum of cross-sectionally ranked components:

```
causal_alpha = w₁ × rank(expr₁) + w₂ × rank(expr₂) + ... + wₖ × rank(exprₖ)
```

Where each `exprᵢ` is a valid OHLCV expression.

---

## 6. Practical Guidelines

### Minimum Data Requirements

- **Observations**: At least 250 × N_stocks for reliable IC estimation
- **Features**: 88+ OHLCV-derived features + multi-horizon targets (5d/10d/20d)
- **Regimes**: At least 4 distinct regimes per method with ≥60 obs each.
  Default: three methods tested simultaneously (volatility, bull/bear, calendar).

### Choosing the Method

| Scenario | Recommended Method |
|----------|-------------------|
| Most cases (recommended) | `ic_selection` (fast, diverse, 88+ features) |
| < 20 variables, need robust skeleton | `pc` |
| 20–50 variables, need effect sizes | `lingam` |
| > 50 variables, GPU available | `notears` |
| Combine PC + LiNGAM | `hybrid` |

### Interpreting Results

- **Diverse parents from different families**: Best case — each component
  captures a genuinely different causal mechanism.
- **Auto vol-gate applied**: The factor is wrapped with a vol filter. This
  means the ATE flips sign in high-volatility regimes. The gated expression
  neutralizes the signal on high-vol stocks.

- **Causal parents with low stability (<0.5)** OR **fail invariance test**:
  These are likely spurious. The edge appeared in some bootstrap runs but not
  consistently, or the effect changes across regimes.

- **Variables that are NOT causal parents but ARE correlated with target**:
  These are mediated or confounded relationships. They may work in backtest
  but are not reliable for live trading.

## 7. Auto Vol-Gating

When the invariance test detects that the ATE flips sign in high-volatility
regimes (consistently negative in low/med vol, positive in high vol), the
`build_causal_alpha.py` script automatically wraps the expression:

```
(base_alpha) * (1 - rank(ts_std(returns(close, 1), 20)))
```

This neutralizes the alpha signal on high-volatility stocks where the causal
relationship breaks down. The gated factor is saved as `causal_factor_gated.csv`
and should be used for backtesting. The `causal_alpha_expression.json` records
`vol_gated: true` when this is applied.

## 8. Multi-Horizon Tuning

Forward return targets at 5d, 10d, and 20d are generated. Running IC selection
on each target and comparing the resulting ATE magnitudes reveals which horizon
has the strongest causal signal. Typically, 20d forward returns produce stronger
and more stable ATE than 5d. The best target is selected for the final alpha.

---

## 7. References

- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction,
  and Search*. MIT Press. (PC algorithm foundation)

- Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006). A linear
  non-Gaussian acyclic model for causal discovery. *JMLR*, 7, 2003-2030.
  (LiNGAM)

- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with
  NO TEARS: Continuous optimization for structure learning. *NeurIPS*.
  (NOTEARS)

- Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference by
  using invariant prediction: identification and confidence intervals.
  *JRSS-B*, 78(5), 947-1012. (Invariance principle)

- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant
  risk minimization. *arXiv:1907.02893*. (IRM — invariant representations)

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.).
  Cambridge University Press. (Do-calculus, SCM framework)
