# Causal Alpha

[简体中文](README.md) | **English**

Discover **regime-invariant alpha factors** from OHLCV data using causal
discovery (PC + LiNGAM + NOTEARS). Unlike traditional correlation-based
factor mining, causal discovery identifies structural relationships between
OHLCV-derived quantities that *cause* future returns — producing factors
that remain stable across market regimes.

This skill is for research purposes only and does not constitute investment advice.

## Core Insight

Traditional factor mining asks:

> "When this OHLCV pattern appeared historically, what happened to returns?"

This is **correlational** reasoning. The problem: financial markets are
non-stationary. Correlations between patterns and returns can and do change
across regimes.

Causal discovery asks a fundamentally different question:

> "Does this OHLCV-derived quantity *cause* future returns through a stable
> structural mechanism?"

If yes, the relationship is **invariant** — it holds in bull, bear, sideways,
and volatile markets alike.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure PandaData for A-share real-time data
cp .env.example .env
# Edit .env with your PANDA_AI_USERNAME and PANDA_AI_PASSWORD
```

## Quick Start

Train/test periods are configured via `data_split` in `config.json` (default: train 2015–2020, test 2021–2025). Factor complexity is controlled by `max_components` (default 3).

### Option 1: PandaData (A-share real-time data)

```bash
# Step 1: Generate train features (split due to 5yr API limit)
python scripts/generate_features.py --use-pandadata --start-date 20150101 --end-date 20191231 --indicator 000300 --output output/run_001/features_2015_2019.csv
python scripts/generate_features.py --use-pandadata --start-date 20200101 --end-date 20201231 --indicator 000300 --output output/run_001/features_2020.csv
# Merge and add forward_return_5d target

# Step 2: Causal discovery (rank-IC selection by default)
python scripts/causal_discovery.py --features output/run_001/features.csv --target forward_return_5d --method ic_selection --output output/run_001/causal_graph.json

# Step 3: Build causal alpha (max 3 components by default)
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features.csv --output output/run_001/causal_factor.csv --expression-output output/run_001/causal_alpha_expression.json --max-components 3

# Step 4: Generate test features & evaluate factor on test period
python scripts/generate_features.py --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output output/run_001/features_test.csv
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features_test.csv --output output/run_001/causal_factor_test.csv --max-components 3

# Step 5: Invariance test + backtest (test period)
python scripts/invariance_test.py --factor output/run_001/causal_factor_test.csv --data-root ../backtest/market_data_2020_2025/ --output output/run_001/invariance_report.json
python scripts/integrate_backtest.py --factor output/run_001/causal_factor_test.csv --factor-column causal_alpha --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output-dir output/run_001/

# Step 6: Generate analysis report
python scripts/generate_report.py --run-dir output/run_001/ --output output/run_001/causal_analysis.md
```

### Option 2: Local Parquet data

```bash
# Step 1: Generate features from local Parquet data
python scripts/generate_features.py \
  --data-root ../backtest/market_data_2015_2019/ \
  --output output/run_001/features.csv

# Steps 2–4: same as above...

# Step 5: Backtest with local data + time range filter
python scripts/integrate_backtest.py \
  --factor output/run_001/causal_factor.csv \
  --factor-column causal_alpha \
  --data-root ../backtest/market_data_2020_2025/ \
  --timespan 20200101 20251231 \
  --output-dir output/run_001/
```

### Advanced Usage

```bash
# Different stock universe
python scripts/generate_features.py \
  --use-pandadata --start-date 20200101 --end-date 20251231 \
  --indicator 000905 \            # CSI 500
  --output output/run_002/features.csv

# Custom backtest parameters
python scripts/integrate_backtest.py \
  --factor output/run_001/causal_factor_test.csv \
  --factor-column causal_alpha \
  --use-pandadata --start-date 20210101 --end-date 20251231 \
  --indicator 000300 \
  --holding-days 10 \             # Rebalance every 10 trading days
  --n-long 100 \                  # Hold 100 stocks
  --cost-bps 3 \                  # 3bps transaction cost
  --output-dir output/run_001/
```

## Stock Universe Codes (PandaData)

| Code | Universe |
|------|----------|
| `000300` | CSI 300 |
| `000905` | CSI 500 |
| `000016` | SSE 50 |
| `000852` | CSI 1000 |

## Output Files

| File | Content |
|------|---------|
| `causal_graph.json` | Discovered causal DAG (edges, ATE) |
| `causal_alpha_expression.json` | OHLCV factor expression + component breakdown |
| `causal_factor.csv` / `causal_factor_test.csv` | Daily factor scores (train/test periods) |
| `invariance_report.json` | Cross-regime invariance test (per-regime ATE with date ranges) |
| `causal_backtest_summary.json` | Backtest metrics (Sharpe, IC, MDD) |
| `causal_analysis.md` | Comprehensive analysis report (18 sections, Mermaid diagrams) |
| `ICs.csv` | Daily multi-horizon Rank IC series |
| `stats.csv` | Portfolio simulation daily NAV |
| `ic_summary.csv` | Multi-horizon IC summary (1d/5d/10d/20d) |

## How It Works

1. **Generate feature pool**: From 6 OHLCV fields, generate 88+ derived features (5 families)
2. **Causal discovery**: Rank-IC selection + diversity (max 1 per family). Excludes future-return
   features. Supports multi-horizon target tuning (5d/10d/20d).
3. **Build alpha**: Top-3 causal parents → valid OHLCV expressions → ATE-weighted → rank normalized.
   Auto vol-gate: if ATE flips in high-vol regimes, wraps with `* (1 - rank(vol))`.
4. **Invariance test**: Three methods simultaneously (volatility, bull/bear, calendar year)
   with combined verdict and per-regime ATE tables.
5. **Backtest**: Built-in engine computes rank IC (1d/5d/10d/20d), long-only portfolio,
   Sharpe, max drawdown, IC autocorrelation.
6. **Analysis report**: ~16-section comprehensive report with signal family classification,
   per-method ATE tables, and dynamic key takeaways.
