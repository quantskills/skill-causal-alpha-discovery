---
name: causal-alpha
description: >-
  Discover causal alpha factors from OHLCV data using causal discovery
  (PC + LiNGAM + NOTEARS), build Structural Causal Models, construct
  regime-invariant factor expressions, and validate through backtesting.
  Produces OHLCV-only alpha factors whose predictive power is grounded in
  causal mechanisms rather than spurious correlations. Use when an agent
  needs to discover stable alpha factors that survive regime changes, test
  whether existing factors are causal or merely correlational, or generate
  alpha factors with formal invariance guarantees on portable agent
  platforms such as Claude Code, Codex, or Codex-style skill systems.
license: GPL-3.0-only
metadata:
  organization: QuantSkills
  organization_url: https://github.com/quantskills
  repository: skill-causal-alpha
  repository_url: https://github.com/quantskills/skill-causal-alpha
  project_type: skill
  collection: factor-generation
  creator: davideliu
  creator_url: https://github.com/davideliu
  maintainer: davideliu
  maintainer_url: https://github.com/davideliu
quantSkills:
  organization: QuantSkills
  organization_url: https://github.com/quantskills
  repository: skill-causal-alpha
  repository_url: https://github.com/quantskills/skill-causal-alpha
  project_type: skill
  collection: factor-generation
  category: factor
  tags:
    - causal-discovery
    - causal-inference
    - alpha-generation
    - factor-discovery
    - regime-invariance
    - structural-causal-model
    - do-calculus
    - ohlcv
    - stock-selection
    - cross-sectional
  platforms:
    - claude-code
    - codex
    - cursor
    - openclaw
  language: zh-en
  status: draft
  validation_level: listed
  maintainer_type: community
  requires: []
summary_zh: 基于IC选择+多样性约束从OHLCV数据中挖掘因果alpha因子，自动检测体制失效并施加波动率门控，支持多目标周期调优，全体制（波动率/牛熊/年度）不变性检验。
summary_en: Discover causal alpha factors from OHLCV data via IC selection with diversity constraints, auto vol-gating from invariance data, multi-horizon target tuning, and three-method regime testing.---
---

# Causal Alpha

Use this skill to discover **causal alpha factors** from OHLCV data. Unlike
traditional correlation-based factor mining, causal discovery identifies
structural relationships between OHLCV-derived quantities that *cause* future
returns — producing factors that are stable across market regimes.

This skill is **self-contained** — no API keys, no external services, no paid
data required. It supports two data sources:
- **PandaData API** — fetch real A-share daily OHLCV data (CSI 300, CSI 500,
  SSE 50, CSI 1000) by setting credentials in `.env`.
- **Local Parquet files** — use pre-downloaded OHLCV data in Parquet format
  for offline/reproducible runs.

It provides:

- **Rank-IC selection** — fast, scalable causal discovery (300K+ observations,
  88+ features). Uses diversity constraints (`diversity_max_per_family`) to
  avoid selecting near-duplicate features differing only by lookback window.
  PC, LiNGAM, and NOTEARS available as alternatives for smaller datasets.
- **Auto vol-gating** — after invariance testing, automatically detects if the
  ATE flips sign in high-volatility regimes and wraps the alpha expression with
  a vol filter: `alpha * (1 - rank(ts_std(returns(close,1), 20)))`.
- **Multi-horizon tuning** — can try forward returns at 5d, 10d, and 20d
  horizons, selecting the target with the strongest causal signal.
- **Three-method regime testing** — tests invariance across volatility regimes,
  bull/bear markets, and calendar years simultaneously (`regime_method: "all"`).
- **Built-in backtest engine** — self-contained cross-sectional backtest with
  rank IC (1d/5d/10d/20d), long-only portfolio, Sharpe, max drawdown, IC
  stability autocorrelation.
- **Comprehensive analysis report** — single Markdown report (~16 sections):
  configuration with train/test split, component breakdown with signal families,
  performance, cumulative return & drawdown, multi-horizon IC, IC autocorrelation,
  causal graph (Mermaid), drawdown analysis, per-method regime ATE tables with
  type labels, and dynamic key takeaways.
- **Configurable train/test split** — `data_split` in `config.json`
  (default: train 2015–2020, test 2021–2025).
- Creator: `davideliu` (`https://github.com/davideliu`).
- Maintainer: `davideliu` for the QuantSkills community.
- Repository: `https://github.com/quantskills/skill-causal-alpha`.
- License: GNU General Public License v3.0 only (`GPL-3.0-only`).
- Scope: OHLCV-only causal alpha factor discovery and regime-invariance
  validation. The skill is not official investment advice, a certified data
  product, or a guarantee of trading performance.

## Core Workflow

### Phase 1: Generate Feature Pool

1. **Generate OHLCV-derived features** — run:
   `python scripts/generate_features.py --use-pandadata --start-date <YYYYMMDD> --end-date <YYYYMMDD> --indicator 000300 --output features.csv`
   Or with local data:
   `python scripts/generate_features.py --data-root <market_data_parquet> --output features.csv`
   This produces 30–50 derivative features using only the 27 allowed functions
   on 6 OHLCV fields. Features span momentum, volatility, volume/flow, price
   pattern, and cross-family interaction families.

### Phase 2: Causal Discovery

2. **Discover causal structure** — run:
   `python scripts/causal_discovery.py --features features.csv --target forward_return_20d --method ic_selection --output causal_graph.json`
   - Default: **rank-IC selection** with diversity (`diversity_max_per_family: 1`).
     Excludes other forward_return variants to prevent look-ahead bias.
   - Try multiple targets (5d/10d/20d) to find the strongest causal signal.
   - Alternative methods: `pc`, `lingam`, `notears`, `hybrid` for deeper analysis.

### Phase 3: Build Causal Alpha

3. **Construct causal alpha** — run:
   `python scripts/build_causal_alpha.py --graph causal_graph.json --features features.csv --output causal_factor.csv --expression-output causal_alpha_expression.json --max-components 3`
   - Limited to 3 components by default (configurable via `max_components`).
   - If `--invariance invariance_report.json` is provided, **auto-applies a vol
     gate** when ATE sign flips in high-volatility regimes.
   - Evaluate on test features separately:
     `python scripts/build_causal_alpha.py --graph causal_graph.json --features features_test.csv --output causal_factor_test.csv --max-components 3`

### Phase 4: Invariance Testing

4. **Test regime invariance** — run:
   `python scripts/invariance_test.py --factor causal_factor_test.csv --data-root <market_data> --regime-method all --output invariance_report.json`
   - Default `regime_method: "all"` runs three independent tests:
     volatility clustering, bull/bear (return quantiles), and calendar year.
   - Produces per-regime ATE tables with type labels and date ranges.
   - Per-method χ² test with p-values and sign consistency.

### Phase 5: Backtest + Auto-Gating

5. **Apply vol-gate and backtest** — run:
   `python scripts/build_causal_alpha.py --graph causal_graph.json --features features_test.csv --output causal_factor_gated.csv --max-components 3 --invariance invariance_report.json`
   `python scripts/integrate_backtest.py --factor causal_factor_gated.csv --factor-column causal_alpha --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output-dir output/run_<ts>/`
   - Auto-gating wraps the expression with `* (1 - rank(ts_std(returns(close,1), 20)))`
     when ATE flips sign in high-vol regimes.
   - Backtest on test period with standalone engine.

### Phase 6: Generate Report

6. **Generate analysis report** — run:
   `python scripts/generate_report.py --run-dir output/run_<ts>/ --output causal_analysis.md`
   - ~16 sections: config, components, performance, IC, IC autocorrelation,
     causal graph, drawdown, combined invariance verdict, per-method ATE tables.

## Output Contract

All files live inside `output/run_<timestamp>/`. The skill root stays clean.

| File | Content |
|------|---------|
| `features.csv` / `features_test.csv` | Generated OHLCV-derived feature pool (train/test) |
| `causal_graph.json` | Discovered causal DAG with edges, ATE, stability scores |
| `causal_alpha_expression.json` | Valid OHLCV factor expression + component breakdown |
| `causal_factor.csv` / `causal_factor_test.csv` | Daily factor scores — train and test periods |
| `invariance_report.json` | Regime-invariance test with per-regime ATE |
| `causal_backtest_summary.json` | Backtest metrics (Sharpe, IC, MDD) |
| `causal_analysis.md` | Comprehensive analysis report (18 sections, Mermaid diagrams) |
| `ICs.csv` | Daily multi-horizon Rank IC series |
| `stats.csv` | Portfolio simulation daily NAV |
| `ic_summary.csv` | Multi-horizon IC summary (1d/5d/10d/20d) |
| `config.json` | Full config snapshot for reproducibility |

## Calling Pattern

```bash
# --- Prerequisites ---
cp .env.example .env  # Edit with PandaData credentials
pip install -r requirements.txt

# --- Full pipeline (train 2015–2020, test 2021–2025) ---

# 1. Generate features (train period 2015-2020, split due to 5yr API limit)
python scripts/generate_features.py --use-pandadata --start-date 20150101 --end-date 20191231 --indicator 000300 --output output/run_001/features_2015_2019.csv
python scripts/generate_features.py --use-pandadata --start-date 20200101 --end-date 20201231 --indicator 000300 --output output/run_001/features_2020.csv
# Merge manually, then add forward_return_5d target

# 2. Causal discovery (rank-IC selection)
python scripts/causal_discovery.py --features output/run_001/features.csv --target forward_return_5d --method ic_selection --output output/run_001/causal_graph.json

# 3. Build causal alpha (train period)
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features.csv --output output/run_001/causal_factor.csv --expression-output output/run_001/causal_alpha_expression.json --max-components 3

# 4. Generate test-period features & evaluate factor on test
python scripts/generate_features.py --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output output/run_001/features_test.csv
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features_test.csv --output output/run_001/causal_factor_test.csv --max-components 3

# 5. Invariance test & backtest (test period)
python scripts/invariance_test.py --factor output/run_001/causal_factor_test.csv --data-root ../backtest/market_data_2020_2025/ --output output/run_001/invariance_report.json
python scripts/integrate_backtest.py --factor output/run_001/causal_factor_test.csv --factor-column causal_alpha --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output-dir output/run_001/

# 6. Generate report
python scripts/generate_report.py --run-dir output/run_001/ --output output/run_001/causal_analysis.md
```

## How This Skill Works (Agent-Native)

1. **You** read `SKILL.md`, `config.json` (especially `data_split`), and `references/causal-discovery.md`.
2. **You** generate features for the train period, run causal discovery (IC-based by default), and build the alpha expression.
3. **You** generate features for the test period and evaluate the alpha on test data.
4. **You** run invariance testing and backtest on the test period.
5. **You** generate the analysis report and interpret the causal graph, performance metrics, and regime stability.

## Reference Files

- `config.json` — all configurable thresholds with `_help`. Key sections: `data_split` (train/test periods), `alpha_construction.max_components` (default 3), `causal_discovery.method` (default ic_selection).
- `references/causal-discovery.md` — full algorithm details for PC, LiNGAM, NOTEARS, and do-calculus.
- `references/how-it-works.md` — conceptual overview of the five-phase pipeline.
- `scripts/generate_features.py` — feature pool generator.
- `scripts/causal_discovery.py` — causal discovery (ic_selection, pc, lingam, notears, hybrid).
- `scripts/build_causal_alpha.py` — alpha expression constructor.
- `scripts/invariance_test.py` — cross-regime stability testing.
- `scripts/integrate_backtest.py` — standalone backtest engine.
- `scripts/generate_report.py` — comprehensive analysis report generator.

## Cross-Agent Use

- Codex and Claude Code can load this folder directly as `$causal-alpha`
  through `SKILL.md`.
- Cursor should use `agents/cursor-rule.mdc` as the project rule adapter and
  keep the full skill folder under `.cursor/skills/causal-alpha`.
- Hermes and OpenClaw should use `agents/portable-loader.md` when they do not
  natively discover `SKILL.md` folders.
- OpenAI-style agent registries can read `agents/openai.yaml` for display name,
  short description, and default invocation prompt.
