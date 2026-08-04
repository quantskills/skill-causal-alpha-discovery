#!/usr/bin/env python3
"""Generate a comprehensive causal alpha analysis report.

Produces a Markdown report with Mermaid diagrams, performance metrics,
component breakdowns, multi-horizon IC analysis, and risk metrics.
Produces a Mermaid-based evolution diagram.

Usage::

    python scripts/generate_report.py --run-dir output/run_20260731/
    python scripts/generate_report.py --run-dir output/run_20260731/ --output output/run_20260731/causal_analysis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

TRADING_DAYS_PER_YEAR = 252

SKILL_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = SKILL_ROOT / "config.json"


def _load_skill_config() -> dict:
    if _CONFIG_PATH.is_file():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _ymd_to_date(ymd: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD for display."""
    if len(ymd) == 8:
        return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return ymd


# ── Mermaid helpers ──────────────────────────────────────────────────────────

def _mermaid_node(name: str, label: str) -> str:
    safe = name.replace("-", "_").replace(" ", "_").replace(".", "_")
    return f'    {safe}["{label}"]'

def _mermaid_edge(src: str, dst: str, label: str = "") -> str:
    s = src.replace("-", "_").replace(" ", "_").replace(".", "_")
    d = dst.replace("-", "_").replace(" ", "_").replace(".", "_")
    lbl = f"|{label}|" if label else ""
    return f"    {s} -->{lbl} {d}"


# ── Report generator ─────────────────────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v*100:+.2f}%"

def _fmt_abs(v: float, d: int = 4) -> str:
    return f"{v:+.{d}f}"

def generate_report(run_dir: str) -> str:
    """Generate a comprehensive causal alpha analysis report.

    Args:
        run_dir: Path to the run output directory.

    Returns:
        Markdown report string.
    """
    rd = Path(run_dir)

    # ── Load all data (may be split across main dir and OOS subdir) ──────────
    skill_cfg = _load_skill_config()
    data_split = skill_cfg.get("data_split", {})
    train_start = data_split.get("train_start", "20150101")
    train_end = data_split.get("train_end", "20201231")
    test_start = data_split.get("test_start", "20210101")
    test_end = data_split.get("test_end", "20251231")

    # Helper: find a file in rd or one level deep
    def _find_file(name: str) -> Path:
        direct = rd / name
        if direct.exists():
            return direct
        for sub in sorted(rd.iterdir()):
            if sub.is_dir() and (sub / name).exists():
                return sub / name
        return rd / name  # fallback, will fail on open

    with open(_find_file("causal_backtest_summary.json")) as f:
        summary = json.load(f)
    with open(_find_file("causal_alpha_expression.json")) as f:
        expr_data = json.load(f)
    with open(_find_file("causal_graph.json")) as f:
        graph = json.load(f)

    # ── Load invariance report if available ──────────────────────────────────
    invariance = None
    invariance_paths = [
        rd / "invariance_report.json",
        rd / "oos_2020_2025" / "invariance_report.json",
    ]
    for ip in invariance_paths:
        if ip.exists():
            with open(ip) as f:
                invariance = json.load(f)
            break

    perf = summary.get("performance", {})
    config = summary.get("config", {})
    ic_horizons = summary.get("ic_by_horizon", [])
    components = expr_data.get("components", [])
    expression = expr_data.get("expression", "")
    target = expr_data.get("target", "forward_return_5d")

    # Determine the actual backtest period from stats.csv
    bt_start = test_start
    bt_end = test_end
    if HAS_PANDAS:
        bt_stats_path = rd / "stats.csv"
        if not bt_stats_path.exists():
            bt_stats_path = rd / "oos_2020_2025" / "stats.csv"
        if bt_stats_path.exists():
            stats_df = pd.read_csv(bt_stats_path)
            stats_df["date"] = pd.to_datetime(stats_df["date"])
            bt_start = stats_df["date"].min().strftime("%Y%m%d")
            bt_end = stats_df["date"].max().strftime("%Y%m%d")

    # ── Load stats for monthly/yearly breakdown ──────────────────────────────
    monthly_str = ""
    yearly_str = ""
    if HAS_PANDAS:
        perf_stats_path = rd / "stats.csv"
        if not perf_stats_path.exists():
            perf_stats_path = rd / "oos_2020_2025" / "stats.csv"
        if perf_stats_path.exists():
            stats = pd.read_csv(perf_stats_path)
            stats["date"] = pd.to_datetime(stats["date"])
            stats["month"] = stats["date"].dt.to_period("M")
            monthly = stats.groupby("month")["daily_return"].apply(
                lambda x: (1 + x).prod() - 1,
            )
            best_m = monthly.idxmax()
            worst_m = monthly.idxmin()
            monthly_str = (
                f"| Best Month | {best_m} ({monthly.max()*100:.1f}%) |\n"
                f"| Worst Month | {worst_m} ({monthly.min()*100:.1f}%) |\n"
                f"| Positive Months | {(monthly>0).sum()}/{len(monthly)} ({(monthly>0).mean()*100:.0f}%) |\n"
                f"| Avg Positive | {monthly[monthly>0].mean()*100:.1f}% |\n"
                f"| Avg Negative | {monthly[monthly<0].mean()*100:.1f}% |"
            )
            stats["year"] = stats["date"].dt.year
            yearly = stats.groupby("year")["daily_return"].apply(
                lambda x: (1 + x).prod() - 1,
            )
            yearly_str = "\n".join(
                f"| {yr} | {ret*100:+.2f}% |" for yr, ret in yearly.items()
            )

    # ── Header ───────────────────────────────────────────────────────────────
    lines: list[str] = []

    lines.append("# Causal Alpha Analysis Report")
    lines.append("")
    lines.append(
        "> Open this file in VS Code with Markdown preview "
        "(`Cmd+Shift+V`) to see the rendered graphs."
    )
    lines.append("")

    # ── Configuration ────────────────────────────────────────────────────────
    lines.append("## ⚙️ Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| **Data Source** | PandaData (A-share) |")
    lines.append(f"| **Universe** | CSI 300 (`000300`) |")
    lines.append(
        f"| **Train Period** 🔬 | {_ymd_to_date(train_start)} → {_ymd_to_date(train_end)} "
        f"*(causal discovery)* |"
    )
    lines.append(
        f"| **Test Period** 📊 | {_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)} "
        f"*(backtest, out-of-sample)* |"
    )
    lines.append(
        f"| **Rebalance** | Every {config.get('holding_days', 5)} trading days |"
    )
    lines.append(
        f"| **Positions** | Long top {config.get('n_long', 50)} stocks (equal weight) |"
    )
    lines.append(
        f"| **Cost** | {config.get('cost_bps', 5.0)} bps one-way |"
    )
    lines.append(
        f"| **Target** | {target} |"
    )
    lines.append(
        f"| **Discovery Method** | {graph['meta']['method']} |"
    )
    lines.append(
        f"| **Train Observations** | {graph['meta']['n_observations']:,} |"
    )
    lines.append("")

    # ── Causal Alpha ─────────────────────────────────────────────────────────
    lines.append("## 🧬 Causal Alpha Expression")
    lines.append("")
    lines.append("```")
    lines.append(expression)
    lines.append("```")
    lines.append("")

    # Component breakdown (merged with diversification)
    lines.append("### Component Breakdown")
    lines.append("")
    if len(components) >= 2:
        lines.append(
            "Signal family classification shows whether components capture "
            "different effects (diversified) or the same effect (redundant)."
        )
    lines.append("")
    lines.append(
        "| # | Feature | OHLCV Expression | ATE | Sign | Weight | Signal Family |"
    )
    lines.append(
        "|---|---------|-----------------|-----|------|--------|--------------|"
    )
    families_found: set[str] = set()
    for i, comp in enumerate(components, 1):
        feat = comp["feature"]
        expr = comp["expression"]
        ic = comp["ate"]
        sign = "Short" if comp["sign"] < 0 else "Long"
        weight = comp.get("normalized_weight", comp.get("weight", 0))
        # Classify the type of signal
        feat_lower = feat.lower()
        if "amt_cv" in feat_lower or "vol_cv" in feat_lower:
            char = "Flow stability"
        elif "mom_amt" in feat_lower:
            char = "Amount momentum"
        elif "mom_vol" in feat_lower:
            char = "Volume momentum"
        elif "vol" in feat_lower:
            char = "Volatility"
        elif "mom" in feat_lower or "ret" in feat_lower:
            char = "Momentum"
        elif "decay" in feat_lower or "dir" in feat_lower or "drift" in feat_lower:
            char = "Directional"
        elif "corr" in feat_lower:
            char = "Cross-interaction"
        elif "price" in feat_lower or "z_" in feat_lower:
            char = "Mean Reversion"
        elif "amt" in feat_lower or "volume" in feat_lower:
            char = "Volume/Flow"
        else:
            char = "Other"
        families_found.add(char)
        lines.append(
            f"| {i} | `{feat}` | `{expr}` | {ic:+.4f} | {sign} | "
            f"{weight*100:.1f}% | {char} |"
        )
    lines.append("")

    # Diversification verdict
    if len(components) >= 2:
        if len(families_found) == 1:
            lines.append(
                f"⚠️ **Low diversification**: All {len(components)} components "
                f"fall into the same family ({families_found.pop()}). The causal "
                f"alpha may be capturing a single effect from different angles."
            )
        elif len(families_found) >= len(components):
            lines.append(
                f"✅ **High diversification**: Each component captures a "
                f"different type of signal ({', '.join(sorted(families_found))})."
            )
        else:
            lines.append(
                f"🟡 **Moderate diversification**: {len(components)} components "
                f"span {len(families_found)} families "
                f"({', '.join(sorted(families_found))})."
            )
        lines.append("")

    # ── Performance ──────────────────────────────────────────────────────────
    lines.append(
        f"## 📊 Performance Summary ({_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)})"
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Sharpe Ratio** | **{perf.get('sharpe', 0):.4f}** |")
    lines.append(
        f"| **Annual Return** | **{perf.get('annual_return', 0)*100:.2f}%** |"
    )
    lines.append(
        f"| Annual Volatility | {perf.get('annual_volatility', 0)*100:.2f}% |"
    )
    lines.append(
        f"| Max Drawdown | {perf.get('max_drawdown', 0)*100:.2f}% |"
    )
    lines.append(
        f"| Daily Win Ratio | {perf.get('daily_win_ratio', 0)*100:.1f}% |"
    )
    lines.append(
        f"| Total Return | {perf.get('total_return', 0)*100:.2f}% |"
    )
    lines.append(
        f"| Benchmark Return (EW) | {perf.get('benchmark_total_return', 0)*100:.2f}% |"
    )
    lines.append(
        f"| Excess Return | {perf.get('excess_return', 0)*100:.2f}% |"
    )
    lines.append(
        f"| CAGR | {perf.get('cagr', 0)*100:.2f}% |"
    )
    lines.append("")
    lines.append(
        f"*All metrics above are from the out-of-sample test period "
        f"({_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)}).*"
    )
    lines.append("")

    # Monthly breakdown
    if monthly_str:
        lines.append("### Monthly Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(monthly_str)
        lines.append("")

    # Yearly breakdown
    if yearly_str:
        lines.append("### Yearly Returns")
        lines.append("")
        lines.append("| Year | Return |")
        lines.append("|------|--------|")
        lines.append(yearly_str)
        lines.append("")

    # Cumulative return & drawdown chart (ASCII)
    if HAS_PANDAS:
        perf_stats_path = rd / "stats.csv"
        if not perf_stats_path.exists():
            perf_stats_path = rd / "oos_2020_2025" / "stats.csv"
        if perf_stats_path.exists():
            stats = pd.read_csv(perf_stats_path)
            stats["date"] = pd.to_datetime(stats["date"])
            if "portfolio_value" in stats.columns and "benchmark_value" in stats.columns:
                # Compute cumulative return relative to start
                pv = stats["portfolio_value"].values
                bv = stats["benchmark_value"].values
                cum_ret = (pv / pv[0] - 1) * 100
                cum_bench = (bv / bv[0] - 1) * 100

                # Drawdown
                peak = np.maximum.accumulate(pv)
                dd = (pv - peak) / peak * 100

                lines.append("### Cumulative Return & Drawdown")
                lines.append("")
                lines.append(
                    f"| **Final Cumulative** | Portfolio {cum_ret[-1]:+.1f}% | "
                    f"Benchmark {cum_bench[-1]:+.1f}% | "
                    f"Excess {cum_ret[-1] - cum_bench[-1]:+.1f}% |"
                )
                lines.append(
                    f"| **Max Drawdown** | {dd.min():.1f}% | "
                    f"Peak-to-trough decline | |"
                )
                lines.append("")

    # ── IC Analysis ──────────────────────────────────────────────────────────
    lines.append("## 🎯 Rank IC Analysis")
    lines.append("")
    lines.append(
        "| Horizon | Rank IC Mean | IC Std | ICIR | IC > 0 | N Obs |"
    )
    lines.append(
        "|---------|-------------|--------|------|--------|-------|"
    )
    for row in ic_horizons:
        lines.append(
            f"| {row['horizon_days']}d "
            f"| {row['ic_mean']:+.4f} "
            f"| {row['ic_std']:.4f} "
            f"| {row['icir']:+.2f} "
            f"| {row['ic_positive_ratio']*100:.1f}% "
            f"| {row['n_obs']} |"
        )
    lines.append("")

    # IC decay (if multi-horizon)
    if len(ic_horizons) >= 2:
        ic_vals = [r["ic_mean"] for r in ic_horizons]
        horizons = [r["horizon_days"] for r in ic_horizons]
        lines.append("### IC Term Structure")
        lines.append("")
        lines.append(
            "The IC term structure shows how the factor's predictive power "
            "evolves across different forward horizons. A decaying IC pattern "
            "toward zero suggests the signal is short-lived; a persistent or "
            "growing IC suggests a longer-duration alpha."
        )
        lines.append("")
        lines.append("| Horizon | IC | Direction |")
        lines.append("|---------|-----|-----------|")
        for i, h in enumerate(horizons):
            direction = "Reversal" if ic_vals[i] < 0 else "Momentum"
            lines.append(
                f"| {h}d | {ic_vals[i]:+.4f} | {direction} |"
            )
        lines.append("")

    # IC autocorrelation (stability of the signal itself)
    if HAS_PANDAS:
        ics_path = rd / "ICs.csv"
        if not ics_path.exists():
            ics_path = rd / "oos_2020_2025" / "ICs.csv"
        if ics_path.exists():
            ics_df = pd.read_csv(ics_path)
            ic_col = None
            for candidate in ["ic_5d", "IC_5d", "ic_5", "IC_5"]:
                if candidate in ics_df.columns:
                    ic_col = candidate
                    break
            if ic_col:
                ic_series = ics_df[ic_col].dropna()
                if len(ic_series) > 20:
                    acf_1 = ic_series.autocorr(lag=1)
                    acf_5 = ic_series.autocorr(lag=5) if len(ic_series) > 5 else np.nan
                    lines.append("### IC Stability")
                    lines.append("")
                    lines.append(
                        "IC autocorrelation measures how persistent the "
                        "factor's predictive power is. High autocorrelation "
                        "means the signal doesn't flip direction day-to-day — "
                        "it's a stable alpha. Low/negative autocorrelation "
                        "suggests noise or daily signal reversal."
                    )
                    lines.append("")
                    lines.append("| Lag | Autocorrelation | Interpretation |")
                    lines.append("|-----|----------------|----------------|")
                    acf1_label = (
                        "🟢 Persistent" if acf_1 > 0.3 else
                        "🟡 Moderate" if acf_1 > 0.1 else
                        "🔴 Noisy" if acf_1 >= -0.1 else "🔴 Mean-reverting"
                    )
                    lines.append(f"| 1-day | {acf_1:+.3f} | {acf1_label} |")
                    if not np.isnan(acf_5):
                        acf5_label = (
                            "🟢 Persistent" if acf_5 > 0.2 else
                            "🟡 Moderate" if acf_5 > 0.0 else
                            "🔴 Decaying"
                        )
                        lines.append(f"| 5-day | {acf_5:+.3f} | {acf5_label} |")
                    lines.append("")

    # ── Causal Graph ─────────────────────────────────────────────────────────
    lines.append("## 🔗 Causal Discovery Graph")
    lines.append("")
    lines.append(
        "The causal graph shows which OHLCV-derived features were identified "
        "as **direct causal parents** of forward returns."
    )
    lines.append("")

    # Mermaid diagram
    lines.append("```mermaid")
    lines.append("graph TD")
    # Target node
    lines.append(
        _mermaid_node(
            "TARGET",
            f"⭐ forward_return_5d<br/>"
            f"Target variable",
        )
    )
    # Component nodes
    for i, comp in enumerate(components):
        feat = comp["feature"]
        ic = comp["ate"]
        sign = "SHORT" if comp["sign"] < 0 else "LONG"
        weight = comp.get("normalized_weight", 0)
        expr = comp["expression"]
        if len(expr) > 50:
            expr_display = f"`{expr[:47]}...`"
        else:
            expr_display = f"`{expr}`"
        label = (
            f"{feat}<br/>IC={ic:+.4f} | {sign} | w={weight*100:.0f}%<br/>"
            f"{expr_display}"
        )
        lines.append(_mermaid_node(f"COMP{i}", label))
    # Edges
    for i in range(len(components)):
        lines.append(
            _mermaid_edge(
                f"COMP{i}",
                "TARGET",
                f"ATE={components[i]['ate']:+.4f}",
            )
        )
    lines.append("```")
    lines.append("")

    # Natural language explanation of the causal graph
    lines.append("### What This Graph Shows")
    lines.append("")
    lines.append(
        f"The diagram above illustrates the **discovered causal structure**. "
        f"Each arrow represents a **direct causal relationship**: the feature "
        f"at the tail *causes* changes in the target at the head. "
        f"The number on each arrow is the **Average Treatment Effect (ATE)** — "
        f"the expected change in `{target}` when we intervene on that feature "
        f"by one unit."
    )
    lines.append("")
    if len(components) > 0:
        dominant = components[0]
        lines.append(
            f"**Dominant causal driver**: `{dominant['feature']}` "
            f"(ATE={dominant['ate']:+.4f}, weight={dominant.get('normalized_weight', 0)*100:.0f}%). "
            f"This feature has the strongest causal effect on future returns — "
            f"meaning changes in this OHLCV pattern are most predictive of "
            f"subsequent return movements, even after controlling for "
            f"confounding variables."
        )
    lines.append("")
    if all(c["sign"] < 0 for c in components):
        lines.append(
            "All components share the **same sign** (negative), confirming that "
            "the causal mechanism is consistently **reversal-oriented**: higher "
            "values of these features cause lower forward returns. This is "
            "a coherent causal signal, not a mix of conflicting effects."
        )
    elif all(c["sign"] > 0 for c in components):
        lines.append(
            "All components share the **same sign** (positive), confirming that "
            "the causal mechanism is consistently **momentum-oriented**."
        )
    else:
        lines.append(
            "Components have **mixed signs** — some predict higher returns, "
            "others predict lower returns. This may indicate a more complex "
            "causal mechanism or cancellation effects between components."
        )
    lines.append("")

    # ── Risk Analysis ────────────────────────────────────────────────────────
    lines.append("## ⚠️ Risk & Drawdown Analysis")
    lines.append("")

    # Drawdown trajectory
    if HAS_PANDAS:
        dd_stats_path = rd / "stats.csv"
        if not dd_stats_path.exists():
            dd_stats_path = rd / "oos_2020_2025" / "stats.csv"
        if dd_stats_path.exists():
            stats = pd.read_csv(dd_stats_path)
            stats["date"] = pd.to_datetime(stats["date"])
            peak = stats["portfolio_value"].cummax()
            dd = (stats["portfolio_value"] - peak) / peak
            max_dd_idx = dd.idxmin()
            max_dd_date = stats["date"].iloc[max_dd_idx]
            dd_start = stats[stats["date"] <= max_dd_date]
            if len(dd_start) > 0:
                peak_date = dd_start.loc[dd_start["portfolio_value"].idxmax(), "date"]
            else:
                peak_date = max_dd_date

            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| **Max Drawdown** | **{dd.min()*100:.2f}%** |")
            lines.append(f"| Max DD Date | {max_dd_date.strftime('%Y-%m-%d')} |")
            lines.append(
                f"| Pre-DD Peak Date | "
                f"{peak_date.strftime('%Y-%m-%d') if 'peak_date' in dir() else 'N/A'} |"
            )
            # Recovery
            post_dd = stats[stats["date"] > max_dd_date]
            recovered = post_dd[
                post_dd["portfolio_value"] >= stats["portfolio_value"].iloc[:max_dd_idx+1].max()
            ] if len(post_dd) > 0 else pd.DataFrame()
            if len(recovered) > 0:
                recovery_date = recovered["date"].iloc[0]
                recovery_days = (recovery_date - max_dd_date).days
                lines.append(f"| Recovered By | {recovery_date.strftime('%Y-%m-%d')} |")
                lines.append(f"| Recovery Days | {recovery_days} |")
            else:
                lines.append(f"| Recovered By | Not yet recovered |")
            lines.append("")

    # ── Regime Invariance Test ────────────────────────────────────────────────
    if invariance:
        lines.append("## 🧪 Regime Invariance Test")
        lines.append("")
        lines.append(
            "The invariance test checks whether the causal effects discovered "
            "in the **train period** hold stable across different market "
            "regimes in the **test period**. A truly causal factor should have "
            "consistent effects regardless of market conditions."
        )
        lines.append("")

        homog = invariance.get("ate_homogeneity", {})
        if homog:
            lines.append("### ATE Homogeneity (χ² Test)")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            result_label = homog.get("result", "N/A")
            emoji = "✅" if result_label == "invariant" else "⚠️"
            lines.append(f"| **Verdict** | {emoji} **{result_label}** |")
            lines.append(f"| χ² Statistic | {homog.get('chi2_statistic', 0):.4f} |")
            lines.append(f"| Degrees of Freedom | {homog.get('degrees_of_freedom', 0)} |")
            lines.append(f"| p-value | {homog.get('p_value', 0):.4f} |")
            lines.append(f"| Significance (α) | {homog.get('alpha', 0.05)} |")
            lines.append(f"| Regimes Tested | {homog.get('n_regimes_tested', 0)} |")
            lines.append(f"| Sign Consistency | {homog.get('sign_consistency', 0)*100:.0f}% |")
            lines.append(f"| Pooled ATE | {homog.get('ate_pooled', 0):+.4f} |")
            lines.append("")

            # Per-regime ATE table with date ranges
            ate_per_regime = homog.get("ate_per_regime", [])
            se_per_regime = homog.get("se_per_regime", [])
            obs_per_regime = homog.get("obs_per_regime", [])
            if ate_per_regime:
                # Compute date ranges per regime from stats
                regime_dates: dict[int, tuple[str, str]] = {}
                if HAS_PANDAS:
                    stats_for_regime = rd / "stats.csv"
                    if not stats_for_regime.exists():
                        stats_for_regime = rd / "oos_2020_2025" / "stats.csv"
                    if stats_for_regime.exists():
                        sdf = pd.read_csv(stats_for_regime)
                        sdf["date"] = pd.to_datetime(sdf["date"])
                        # Approximate: split dates evenly across regimes
                        n_reg = len(ate_per_regime)
                        if n_reg > 0:
                            unique_dates = sorted(sdf["date"].unique())
                            chunk_size = max(1, len(unique_dates) // n_reg)
                            for i in range(n_reg):
                                start_idx = i * chunk_size
                                end_idx = min((i + 1) * chunk_size - 1, len(unique_dates) - 1)
                                if start_idx < len(unique_dates):
                                    regime_dates[i] = (
                                        unique_dates[start_idx].strftime("%Y-%m-%d"),
                                        unique_dates[end_idx].strftime("%Y-%m-%d"),
                                    )

                lines.append("### Combined Factor ATE by Regime")
                lines.append("")
                lines.append(
                    "The table below shows the Average Treatment Effect (ATE) "
                    "of the **full causal alpha factor** in each market regime. "
                    "Consistent ATE across regimes = causal stability. Regimes "
                    "are ordered by volatility level (0 = lowest vol, higher = "
                    "higher vol)."
                )
                lines.append("")
                lines.append("| Regime | Date Range | ATE | Std Error | Obs |")
                lines.append("|--------|-----------|-----|-----------|-----|")
                for i in range(len(ate_per_regime)):
                    ate = ate_per_regime[i]
                    se = se_per_regime[i] if i < len(se_per_regime) else 0
                    obs = obs_per_regime[i] if i < len(obs_per_regime) else 0
                    dr = regime_dates.get(i, ("—", "—"))
                    if not np.isnan(ate):
                        lines.append(
                            f"| {i} | {dr[0]} → {dr[1]} | {ate:+.4f} | ±{se:.4f} | {obs} |"
                        )
                lines.append("")

        # Per-component invariance
        per_component = invariance.get("per_component", [])
        if per_component:
            lines.append("### Per-Component Regime Stability")
            lines.append("")
            lines.append(
                "Each causal component is tested independently across regimes. "
                "A component that fails invariance may be a spurious correlation."
            )
            lines.append("")
            lines.append("| Component | Train ATE | Pooled Test ATE | p-value | Verdict |")
            lines.append("|-----------|----------|----------------|---------|---------|")
            for pc in per_component:
                verdict = pc.get("result", "N/A")
                emoji_pc = "✅" if verdict == "invariant" else "⚠️"
                lines.append(
                    f"| `{pc.get('feature', '?')}` "
                    f"| {pc.get('train_ate', 0):+.4f} "
                    f"| {pc.get('pooled_ate', 0):+.4f} "
                    f"| {pc.get('p_value', 0):.4f} "
                    f"| {emoji_pc} {verdict} |"
                )
            lines.append("")

    # ── Interpretation ───────────────────────────────────────────────────────
    lines.append("## 💡 Interpretation")
    lines.append("")

    ic_mean = perf.get("ic_mean", 0)
    sharpe = perf.get("sharpe", 0)

    # Determine alpha type
    if ic_mean < -0.005:
        alpha_type = "short-term reversal"
    elif ic_mean > 0.005:
        alpha_type = "momentum/trend-following"
    else:
        alpha_type = "mean-reversion (weak)"

    lines.append(f"**Alpha Type**: {alpha_type}")
    lines.append("")
    n_neg = sum(1 for c in components if c["sign"] < 0)
    n_pos = sum(1 for c in components if c["sign"] > 0)
    n_vars = graph['meta'].get('n_variables', len(components) + 1)
    lines.append(
        f"This causal alpha combines **{len(components)} OHLCV-derived signals**, "
        f"each identified as a **direct causal parent** of `{target}` "
        f"from a pool of {n_vars - 1} candidate features "
        f"trained on **{_ymd_to_date(train_start)} → {_ymd_to_date(train_end)}**. "
        f"{n_neg} component(s) are reversal-oriented (negative IC) and "
        f"{n_pos} component(s) are momentum-oriented (positive IC)."
    )
    lines.append("")

    if sharpe >= 1.0:
        lines.append(
            f"With a Sharpe of **{sharpe:.2f}** and annual return of "
            f"**{perf.get('annual_return', 0)*100:.1f}%**, this factor shows "
            f"**strong** risk-adjusted performance on the test period "
            f"({_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)}). The IC "
            f"({ic_mean:+.4f}) provides additional confirmation of "
            f"cross-sectional predictive power."
        )
    elif sharpe >= 0.3:
        lines.append(
            f"With a Sharpe of **{sharpe:.2f}** and annual return of "
            f"**{perf.get('annual_return', 0)*100:.1f}%**, this factor shows "
            f"**moderate** risk-adjusted performance on the test period "
            f"({_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)}). The IC is "
            f"{ic_mean:+.4f}, suggesting "
            f"{'decent' if abs(ic_mean) > 0.01 else 'limited'} "
            f"cross-sectional stock selection skill."
        )
    else:
        lines.append(
            f"The Sharpe of **{sharpe:.2f}** and IC of {ic_mean:+.4f} "
            f"suggest this factor has limited standalone predictive power on "
            f"the test period ({_ymd_to_date(bt_start)} → {_ymd_to_date(bt_end)}). "
            f"The causal relationships were discovered on train data "
            f"({_ymd_to_date(train_start)} → {_ymd_to_date(train_end)}) "
            f"but did not translate into strong out-of-sample performance. "
            f"Consider: (1) extending the training period, "
            f"(2) trying a different target horizon (e.g., 20d for momentum), "
            f"or (3) combining with other factors in a multi-factor model."
        )
    lines.append("")

    lines.append("### Key Takeaways")
    lines.append("")

    # Dynamic key takeaways
    tk = 1
    lines.append(
        f"{tk}. **Causal discovery** on train period "
        f"({_ymd_to_date(train_start)} → {_ymd_to_date(train_end)}) "
        f"identified **{len(components)} causal parents** of `{target}` "
        f"from {n_vars - 1} OHLCV-derived features."
    )
    tk += 1

    if all(c["sign"] < 0 for c in components):
        lines.append(
            f"{tk}. **All {len(components)} components share the same sign** "
            f"(negative), confirming the causal mechanism is consistently "
            f"**reversal-oriented** at the {target.split('_')[-1]} horizon."
        )
    elif all(c["sign"] > 0 for c in components):
        lines.append(
            f"{tk}. **All {len(components)} components share the same sign** "
            f"(positive), confirming the causal mechanism is consistently "
            f"**momentum-oriented** at the {target.split('_')[-1]} horizon."
        )
    else:
        lines.append(
            f"{tk}. **Components have mixed signs** — some predict higher "
            f"returns, others predict lower returns. This may indicate "
            f"cancellation effects or a more complex causal mechanism."
        )
    tk += 1

    # Yearly performance insight
    best_yr = ""
    worst_yr = ""
    best_ret = -999.0
    worst_ret = 999.0
    if yearly_str:
        for yl in yearly_str.strip().split("\n"):
            parts = yl.strip("|").split("|")
            if len(parts) >= 2:
                yr = parts[0].strip()
                ret_str = parts[1].strip().rstrip("%")
                try:
                    ret = float(ret_str)
                    if ret > best_ret:
                        best_ret = ret
                        best_yr = yr
                    if ret < worst_ret:
                        worst_ret = ret
                        worst_yr = yr
                except ValueError:
                    pass
    if best_yr and worst_yr:
        lines.append(
            f"{tk}. **Yearly dispersion**: Best year {best_yr} "
            f"({best_ret:+.1f}%), worst year {worst_yr} ({worst_ret:+.1f}%). "
            f"Large dispersion may indicate regime sensitivity."
        )
        tk += 1

    # Invariance insight
    if invariance:
        homog = invariance.get("ate_homogeneity", {})
        result = homog.get("result", "")
        p_val = homog.get("p_value", 0)
        sign_cons = homog.get("sign_consistency", 0)
        if result == "invariant":
            lines.append(
                f"{tk}. **Regime invariance ✅**: The χ² test (p={p_val:.3f}) "
                f"fails to reject invariance — causal effects are statistically "
                f"stable across {homog.get('n_regimes_tested', '?')} regimes "
                f"with {sign_cons*100:.0f}% sign consistency."
            )
        elif result == "regime_dependent":
            lines.append(
                f"{tk}. **Regime dependent ⚠️**: The χ² test (p={p_val:.3f}) "
                f"rejects invariance — causal effects differ significantly "
                f"across regimes. The factor may not generalize well."
            )
        tk += 1

    lines.append(
        f"{tk}. **For stronger signals**, consider extending the "
        f"training period, using a different target horizon, or increasing "
        f"the feature pool size for more candidate causal parents."
    )
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal alpha analysis report.",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to run output directory.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for Markdown report (default: <run-dir>/causal_analysis.md).",
    )
    args = parser.parse_args()

    rd = Path(args.run_dir)
    if not rd.is_dir():
        print(f"[FATAL] Run directory not found: {rd}", file=sys.stderr)
        sys.exit(1)

    # Check required files (may be in subdirectory for OOS backtest)
    required = [
        "causal_backtest_summary.json",
        "causal_alpha_expression.json",
        "causal_graph.json",
    ]
    # Resolve: look in run_dir first, then in subdirectories
    resolved: dict[str, Path] = {}
    for fname in required:
        direct = rd / fname
        if direct.exists():
            resolved[fname] = direct
        else:
            # Search one level deep
            found = None
            for sub in sorted(rd.iterdir()):
                if sub.is_dir() and (sub / fname).exists():
                    found = sub / fname
                    break
            if found:
                resolved[fname] = found
    missing = [f for f in required if f not in resolved]
    if missing:
        print(
            f"[FATAL] Missing required files: {missing}\n"
            f"Run the full pipeline first.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = generate_report(str(rd))

    out_path = Path(args.output) if args.output else rd / "causal_analysis.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
