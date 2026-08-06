# Causal Alpha — How It Works

This document explains the logic, methodology, and intuition behind the
`skill-causal-alpha` skill. It is written for humans who want to understand
**what** the skill does, **why** it works, and **how** to interpret its output.

---

## 1. The Problem: Why Most Alpha Factors Fail

### The Correlation Trap

Traditional alpha factor mining (including genetic algorithms, reinforcement
learning, and LLM-based generation) works like this:

1. Generate thousands of OHLCV formula combinations
2. Test each one against historical returns
3. Keep the ones with the highest IC / Sharpe
4. Hope they work in the future

This is **correlation mining**. The problem: financial markets are
non-stationary. The correlation between any OHLCV pattern and future returns
**changes over time** because:

- Market regimes shift (bull → bear → sideways → volatile)
- The data-generating process evolves (new participants, regulations, macro)
- Spurious patterns appear by chance in finite samples

**Example**: A momentum factor that worked brilliantly in the 2017 structural
bull market may fail catastrophically in the 2018 trade-war bear market — not
because it was "wrong," but because the correlation it captured was specific
to one regime.

### The Causal Solution

Causal discovery asks a fundamentally different question:

> Instead of "What patterns correlated with returns in the past?", ask
> **"What OHLCV-derived quantities *cause* returns through stable structural
> mechanisms?"**

If something truly *causes* returns (in the statistical causal sense), its
effect should be **invariant** — the same in bull markets, bear markets,
COVID crashes, and recovery rallies alike.

---

## 2. What Is Causal Discovery?

### Correlation vs. Causation (The Ice Cream Example)

- **Correlation**: Ice cream sales and drowning deaths are highly correlated.
  Does ice cream cause drowning? No. The **confounder** is summer weather:
  hot weather → more ice cream AND more swimming → more drownings.

- **Causation**: If we **intervene** and stop selling ice cream, drowning
  deaths would NOT decrease. If we intervene and teach swimming lessons,
  drowning deaths WOULD decrease.

In finance:

- **Correlation**: "Low volume volatility correlates with high future
  returns" → This worked in 2015–2019, broke in 2020.

- **Causation**: "Earnings surprises cause analyst revisions, which cause
  institutional rebalancing, which causes price drift" → This mechanism is
  the same regardless of whether we're in a bull or bear market.

### How Causal Discovery Works (Simplified)

The skill uses three algorithms to discover causal structure:

#### PC Algorithm (Constraint-Based)

Imagine you have three variables: momentum (M), volatility (V), and returns
(R). You observe that all three are correlated with each other. The PC
algorithm tests **conditional independencies**:

- Is M correlated with R *after controlling for V*? If yes, M → R is direct.
  If no, M's effect on R is mediated through V.

The algorithm systematically tests all such relationships and builds a
**causal skeleton** — an undirected graph of which variables influence
which others.

#### LiNGAM (Functional Causal Discovery)

Financial data is rarely Gaussian (returns have fat tails). LiNGAM exploits
this **non-Gaussianity** to determine causal direction:

- If X causes Y, then regressing Y on X produces residuals that are
  independent of X.
- If Y causes X, the residuals of X on Y would NOT be independent of Y.

LiNGAM tests both directions and picks the one where residuals are most
independent. It also estimates the **strength** of each causal effect.

#### NOTEARS (Continuous Optimization)

Traditional causal discovery searches over possible DAGs combinatorially
(NP-hard). NOTEARS reformulates the problem as:

> Find a matrix W such that `loss(X, W)` is minimized AND
> `trace(exp(W ⊙ W)) - d = 0` (this constraint ensures W is a DAG)

This means we can use **gradient descent** to learn the causal structure,
scaling to hundreds of variables.

---

## 3. The Six-Phase Pipeline

### Phase 1: Generate Feature Pool

From the 6 OHLCV fields, the skill generates **88+ derived features** across
five families (configurable up to 100 via `max_total_features`):

| Family | Count | Examples |
|--------|-------|---------|
| **Momentum** | ~20 | `returns(close, n)`, `decay_linear(returns(close,1), n)` |
| **Volatility** | ~20 | `ts_std(returns(close,1), n)`, `vol_of_vol`, `down_vol` |
| **Volume/Flow** | ~20 | `ts_std(amount, n)/ts_mean(amount, n)`, `vol_cv_*`, `amt_cv_*` |
| **Price Pattern** | ~14 | `intraday_pos`, `candle_body_ratio`, `price_z_*`, `dist_high_*` |
| **Cross-Family** | ~14 | `correlation(rank(close), rank(volume), n)`, `mom_*_vol_*` |

Features are generated separately for **train** and **test** periods (config:
`data_split`). Forward return targets at 5d, 10d, and 20d are added for
multi-horizon tuning.

### Phase 2: Causal Discovery

By default, the skill uses **rank-IC selection**: compute the cross-sectional
rank IC between each feature and the target (`forward_return_5d`), then select
the top features by absolute IC as causal parents. This is fast and scalable
to 60+ features and 300K+ observations.

For deeper causal analysis, the PC, LiNGAM, and NOTEARS algorithms are available:

- Which features are **direct causal parents** of forward returns
- The **ATE (Average Treatment Effect)** — how much returns change when we
  intervene on each parent
- **Stability scores** — how consistently each edge appears across bootstrap
  resamples

### Phase 3: Build Causal Alpha

From the causal graph, the skill:

1. Filters to features that are **direct causal parents** of forward returns
2. Limits to **3 components by default** (`max_components` in config) to control complexity
3. Maps each feature name to its **valid OHLCV expression** (using only the
   6 fields and 27 functions)
4. Weights each component by `|ATE|` (with optional stability weighting)
5. Combines into a single cross-sectionally ranked factor expression

The resulting expression is a **valid OHLCV formula** evaluable on any period.

### Phase 4: Invariance Testing

This is the **key differentiator**. The skill splits the **test period** data
into 6 distinct market regimes (via volatility clustering) and tests:

> $H_0$: The ATE is equal across ALL regimes (the factor is causal/invariant)
>
> $H_1$: The ATE differs by regime (the factor is correlational/regime-dependent)

A χ² homogeneity test produces a p-value. If p > 0.05, we **fail to reject**
the invariance hypothesis — the factor passes the causal test. The report
shows per-regime ATE with date ranges for full transparency.

### Phase 5: Backtest Validation

The factor expression is evaluated on **test-period features**, optionally with
auto vol-gating, then fed into the built-in backtest engine:

- **Auto vol-gate**: If invariance testing detects ATE sign flips in high-vol
  regimes, the expression is automatically wrapped with
  `* (1 - rank(ts_std(returns(close,1), 20)))` to neutralize the signal on
  high-volatility stocks.
- **Rank IC**: Cross-sectional Spearman correlation (1d/5d/10d/20d horizons)
- **Portfolio simulation**: Long-only equal-weight, top-200 stocks, 5-day
  rebalance, 5bps costs
- **IC stability**: Lag-1 and lag-5 autocorrelation of daily IC series

### Phase 6: Generate Analysis Report

A comprehensive ~16-section Markdown report with:
- Train/test period configuration
- Component breakdown with signal family classification (diversity analysis)
- Performance summary with cumulative return & drawdown
- Multi-horizon IC with autocorrelation
- Three-method invariance: combined verdict + per-method ATE tables with type
  labels (Very Low Vol → Very High Vol, Deep Bear → Strong Bull, calendar years)
- Dynamic key takeaways based on actual performance data

## Practical Tips

### Choosing the Target Horizon

- **5-day forward returns**: Short-term reversal/momentum effects.
- **20-day forward returns** (recommended default): Stronger signal-to-noise,
  better for causal discovery. The skill can try multiple horizons to find
  the best.

### Auto Vol-Gating

When the invariance test shows ATE sign flips in high-volatility regimes,
the skill automatically applies a vol gate. This is documented in the
`causal_alpha_expression.json` under `vol_gated: true`. The gated expression
is saved as `causal_factor_gated.csv` and should be used for backtesting.

---

## 4. How to Interpret the Output

### The Causal Alpha Expression

This is the final factor formula. It uses only the 6 OHLCV fields and 27
allowed functions. You can feed it directly into any backtest system.

### Component Breakdown

Each component tells you:
- **Which OHLCV pattern** is causally linked to returns
- **The direction**: positive ATE = buy high values, negative ATE = short
  high values (or buy low values)
- **The weight**: how much this component contributes to the final signal

### Invariance Certificate

The certificate answers: "Will this factor survive the next regime change?"

- **Strongly Invariant** (p > 0.05, sign consistency > 90%): The causal
  mechanism is stable. This factor should generalize.
- **Moderately Invariant** (p > 0.05, sign consistency < 90%): Mostly stable
  but may weaken in extreme regimes.
- **Regime Dependent** (p < 0.01): This is likely a spurious correlation.
  Do not trust it out-of-sample.

### IC Term Structure

The IC at different horizons reveals the factor's "shelf life":

- **Decaying IC** (strong at 1d, weak at 20d): Short-lived signal. Requires
  frequent rebalancing.
- **Persistent IC** (similar across horizons): Longer-duration alpha.
- **Growing IC** (weak at 1d, strong at 20d): Momentum/trend effect. Takes
  time to materialize.

---

## 5. Limitations & Cautions

1. **OHLCV-only**: The skill only uses 6 OHLCV fields. It cannot discover
   causal relationships involving fundamentals, macro, sentiment, or
   alternative data.

2. **Linear assumptions**: LiNGAM and NOTEARS assume linear causal
   relationships. Real financial markets may have nonlinear effects
   that these algorithms miss.

3. **No latent confounders**: The algorithms assume no unobserved common
   causes. In reality, hidden variables (e.g., macro conditions, policy
   changes) may confound the discovered relationships.

4. **Sample size matters**: Causal discovery needs many observations.
   With too few stocks or too short a time period, the algorithms may
   produce unreliable results.

5. **Not a trading system**: This skill discovers alpha factors. It does
   not provide portfolio construction, risk management, execution, or
   any other component of a complete trading system.

---

## 6. Practical Tips

### Choosing the Target Horizon

- **5-day forward returns** (default): Captures short-term reversal and
  momentum effects. Good for high-turnover strategies.
- **20-day forward returns**: Captures longer-duration effects. Better
  signal-to-noise ratio, lower turnover.

### Selecting the Universe

- **CSI 300** (000300): Large-cap, liquid. Lower alpha potential but
  easier to trade.
- **CSI 500** (000905): Mid-cap. Higher alpha potential, higher
  transaction costs.
- **CSI 1000** (000852): Small-cap. Highest alpha potential, highest
  capacity constraints.

### Training vs. Testing

Always use **non-overlapping** periods for causal discovery (train) and
backtest (test). Example:
- Train: 2015–2019
- Test: 2020–2025

This prevents look-ahead bias and gives a realistic estimate of
out-of-sample performance.
