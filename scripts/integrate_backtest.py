#!/usr/bin/env python3
"""Built-in cross-sectional backtest for causal alpha factors.

Provides a self-contained backtest engine — no dependency on any other skill or external tool.

Computes:
  - Cross-sectional rank IC (daily, multi-horizon)
  - Long-only equal-weight portfolio simulation
  - Sharpe ratio, max drawdown, annual return, win ratio
  - IC decay comparison vs. baseline correlational factors

Usage::

    # From local Parquet data
    python scripts/integrate_backtest.py --factor causal_factor.csv --factor-column causal_alpha --data-root ../backtest/market_data_2020_2025/ --output-dir output/run_001/
    python scripts/integrate_backtest.py --factor causal_factor.csv --factor-column causal_alpha --data-root ./market_data/ --timespan 20200101 20251231 --output-dir ./output/
    python scripts/integrate_backtest.py --factor causal_factor.csv --factor-column causal_alpha --data-root ./market_data/ --baseline-ic ../backtest/performance_summary_long_only.csv --output-dir ./output/

    # From PandaData API (A-share real-time)
    python scripts/integrate_backtest.py --factor causal_factor.csv --factor-column causal_alpha --use-pandadata --start-date 20200101 --end-date 20251231 --indicator 000300 --output-dir ./output/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = SKILL_ROOT / "config.json"

def _load_config() -> dict:
    if _CONFIG_PATH.is_file():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}

_config = _load_config()
_bt_cfg = _config.get("backtest", {})

DEFAULT_COST_BPS: float = _bt_cfg.get("cost_bps", 5.0)
DEFAULT_HOLDING_DAYS: int = _bt_cfg.get("holding_days", 5)
DEFAULT_LONGX: int = _bt_cfg.get("longx", 200)
TRADING_DAYS_PER_YEAR: int = 252


# ═══════════════════════════════════════════════════════════════════════════════
# OHLCV Data Loader (self-contained, same logic as generate_features.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_market_data_from_pandadata(
    start_date: str,
    end_date: str,
    indicator: str = "000300",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV fields from PandaData API.

    Args:
        start_date: Start date as YYYYMMDD.
        end_date: End date as YYYYMMDD.
        indicator: Stock universe code.

    Returns:
        Dict mapping field name → (date × ticker) DataFrame.
    """
    from pandadata_loader import (
        fetch_ohlcv_from_pandadata,
        pandadata_to_pivoted_fields,
    )

    print(
        f"Fetching data from PandaData: {start_date}–{end_date}, "
        f"indicator={indicator}",
        file=sys.stderr,
    )
    raw_df = fetch_ohlcv_from_pandadata(
        start_date=start_date,
        end_date=end_date,
        indicator=indicator,
    )
    if raw_df is None:
        return {}

    return pandadata_to_pivoted_fields(raw_df)


def load_market_data(data_root: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV fields from Parquet market data root.

    Returns dict mapping field name → (date × ticker) DataFrame.
    """
    root = Path(data_root)
    fields: dict[str, pd.DataFrame] = {}

    file_map = {
        "trade_price.parquet": "close",
        "open_price.parquet": "open",
    }

    for fname, field in file_map.items():
        fpath = root / fname
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        if "date" not in df.columns or "ticker" not in df.columns:
            continue
        val_col = df.columns.difference(["date", "ticker"])[0]
        pivot = df.pivot_table(
            index="date", columns="ticker", values=val_col, aggfunc="first",
        )
        pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
        pivot = pivot.sort_index()
        fields[field] = pivot

    return fields


def compute_forward_returns(
    prices: pd.DataFrame,
    horizon_days: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """Compute forward returns at multiple horizons.

    Args:
        prices: (date × ticker) close price DataFrame.
        horizon_days: List of forward horizons in trading days.

    Returns:
        Dict mapping horizon → (date × ticker) forward return DataFrame.
        Forward returns are shifted so that fwd_ret[t] = (P[t+h] - P[t]) / P[t].
    """
    if horizon_days is None:
        horizon_days = [1, 5, 10, 20]

    result = {}
    for h in horizon_days:
        fwd = prices.pct_change(h).shift(-h)
        result[h] = fwd
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Sectional Rank IC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily cross-sectional rank IC.

    For each date, computes Spearman rank correlation between factor
    values and forward returns across all stocks active on that date.

    Args:
        factor: (date × ticker) factor values.
        forward_returns: (date × ticker) forward returns.

    Returns:
        Series of daily IC values indexed by date.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_tickers = factor.columns.intersection(forward_returns.columns)

    if len(common_dates) < 5 or len(common_tickers) < 5:
        print(
            f"[WARN] Only {len(common_dates)} dates × {len(common_tickers)} tickers "
            f"in common. IC may be unreliable.",
            file=sys.stderr,
        )
        return pd.Series(dtype=float)

    f_sub = factor.loc[common_dates, common_tickers]
    r_sub = forward_returns.loc[common_dates, common_tickers]

    ic_values: list[float] = []
    ic_dates: list[pd.Timestamp] = []

    for date in common_dates:
        f_row = f_sub.loc[date].dropna()
        r_row = r_sub.loc[date].dropna()

        # Align tickers
        aligned = f_row.index.intersection(r_row.index)
        if len(aligned) < 10:
            continue

        f_vals = f_row.loc[aligned].rank(pct=True).values
        r_vals = r_row.loc[aligned].values

        if np.std(f_vals) == 0 or np.std(r_vals) == 0:
            continue

        corr = np.corrcoef(f_vals, r_vals)[0, 1]
        if not np.isnan(corr):
            ic_values.append(float(corr))
            ic_dates.append(date)

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name="rank_ic")


# ═══════════════════════════════════════════════════════════════════════════════
# Long-Only Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_long_only_portfolio(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    n_long: int = DEFAULT_LONGX,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Simulate a long-only equal-weight portfolio.

    On each rebalance date:
      1. Rank stocks by factor value (higher = better)
      2. Buy top n_long stocks equally weighted
      3. Hold for holding_days, then rebalance
      4. Transaction costs applied at each rebalance

    Args:
        factor: (date × ticker) factor scores.
        prices: (date × ticker) close prices.
        holding_days: Rebalance interval in trading days.
        n_long: Number of stocks to hold long.
        cost_bps: One-way transaction cost in basis points.

    Returns:
        DataFrame with columns: date, portfolio_value, daily_return,
        benchmark_value.
    """
    common_dates = factor.index.intersection(prices.index)
    common_tickers = factor.columns.intersection(prices.columns)

    if len(common_dates) < 20 or len(common_tickers) < n_long:
        print(
            f"[WARN] Insufficient data for portfolio: {len(common_dates)} dates, "
            f"{len(common_tickers)} tickers (need ≥{n_long}).",
            file=sys.stderr,
        )
        return pd.DataFrame()

    f_sub = factor.loc[common_dates, common_tickers].sort_index()
    p_sub = prices.loc[common_dates, common_tickers].sort_index()

    dates = f_sub.index.tolist()
    daily_returns = p_sub.pct_change().shift(-1)

    # Rebalance schedule
    rebalance_dates = dates[::holding_days]
    if rebalance_dates[-1] != dates[-1]:
        rebalance_dates = list(rebalance_dates) + [dates[-1]]

    cost_rate = cost_bps / 10000

    portfolio_value = 1.0
    benchmark_value = 1.0
    holdings: set = set()
    prev_holdings: set = set()

    records: list[dict[str, Any]] = []

    for i, rebal_date in enumerate(rebalance_dates[:-1]):
        next_rebal = rebalance_dates[i + 1]
        rebal_idx = dates.index(rebal_date)
        next_idx = dates.index(next_rebal)

        f_row = f_sub.loc[rebal_date].dropna()
        if len(f_row) < n_long:
            continue

        top_stocks = set(f_row.nlargest(n_long).index.tolist())
        prev_holdings = holdings
        holdings = top_stocks

        sold = prev_holdings - holdings
        bought = holdings - prev_holdings
        turnover = (len(sold) + len(bought)) / max(n_long, 1)
        txn_cost = turnover * cost_rate

        period_dates = dates[rebal_idx:next_idx]
        for d_idx, date in enumerate(period_dates):
            if d_idx == 0:
                portfolio_value *= (1 - txn_cost)

            dr = daily_returns.loc[date]
            holding_rets = dr[list(holdings & set(dr.dropna().index))]
            port_ret = holding_rets.mean() if len(holding_rets) > 0 else 0.0
            portfolio_value *= (1 + port_ret)

            bench_rets = dr.dropna()
            bench_ret = bench_rets.mean() if len(bench_rets) > 0 else 0.0
            benchmark_value *= (1 + bench_ret)

            records.append({
                "date": date,
                "portfolio_value": round(portfolio_value, 6),
                "daily_return": round(port_ret, 6),
                "benchmark_value": round(benchmark_value, 6),
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    portfolio_df: pd.DataFrame,
    ic_series: pd.Series,
) -> dict[str, Any]:
    """Compute standard performance metrics.

    Args:
        portfolio_df: Portfolio simulation output.
        ic_series: Daily rank IC series.

    Returns:
        Dict of metrics.
    """
    metrics: dict[str, Any] = {}

    # IC metrics
    if len(ic_series) > 0:
        metrics["ic_mean"] = round(float(ic_series.mean()), 6)
        metrics["ic_std"] = round(float(ic_series.std()), 6)
        metrics["icir"] = round(
            float(ic_series.mean() / ic_series.std().clip(min=1e-8)), 4,
        )
        metrics["ic_positive_ratio"] = round(float((ic_series > 0).mean()), 4)
        metrics["n_ic_days"] = len(ic_series)

    # Portfolio metrics
    if portfolio_df is not None and len(portfolio_df) > 0:
        daily_rets = portfolio_df["daily_return"].dropna().values
        port_vals = portfolio_df["portfolio_value"].values
        bench_vals = portfolio_df["benchmark_value"].values

        if len(daily_rets) >= 20:
            ann_ret = float(np.mean(daily_rets)) * TRADING_DAYS_PER_YEAR
            ann_vol = float(np.std(daily_rets, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            metrics["sharpe"] = round(ann_ret / ann_vol, 4) if ann_vol > 0 else 0.0
            metrics["annual_return"] = round(ann_ret, 4)
            metrics["annual_volatility"] = round(ann_vol, 4)

            peak = np.maximum.accumulate(port_vals)
            drawdowns = (port_vals - peak) / peak
            metrics["max_drawdown"] = round(float(drawdowns.min()), 4)

            metrics["daily_win_ratio"] = round(float((daily_rets > 0).mean()), 4)
            metrics["total_return"] = round(float(port_vals[-1] / port_vals[0] - 1), 4)
            metrics["benchmark_total_return"] = round(
                float(bench_vals[-1] / bench_vals[0] - 1), 4,
            )
            metrics["excess_return"] = round(
                metrics["total_return"] - metrics["benchmark_total_return"], 4,
            )

            n_years = len(daily_rets) / TRADING_DAYS_PER_YEAR
            if n_years > 0:
                metrics["cagr"] = round(
                    float((port_vals[-1] / port_vals[0]) ** (1 / n_years) - 1), 4,
                )

            metrics["n_days"] = len(daily_rets)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Causal vs. Correlational Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_to_baseline(
    causal_metrics: dict[str, Any],
    baseline_ic_path: str | None = None,
) -> dict[str, Any]:
    """Compare causal factor metrics against baseline correlational factors."""
    comparison: dict[str, Any] = {"causal_factor": causal_metrics}

    if baseline_ic_path and Path(baseline_ic_path).exists():
        try:
            baseline = pd.read_csv(baseline_ic_path)
            if "ic_mean_pct" in baseline.columns:
                baseline_ic = baseline["ic_mean_pct"].dropna() / 100
                comparison["baseline_ic_mean"] = round(float(baseline_ic.mean()), 6)
                comparison["baseline_ic_std"] = round(float(baseline_ic.std()), 6)
                comparison["baseline_n_factors"] = len(baseline_ic)

                causal_ic = causal_metrics.get("ic_mean", 0)
                if comparison["baseline_ic_mean"] != 0:
                    comparison["ic_advantage"] = round(
                        float(causal_ic) / comparison["baseline_ic_mean"] - 1, 4,
                    )

            if "sharpe" in baseline.columns:
                baseline_sharpe = baseline["sharpe"].dropna()
                comparison["baseline_sharpe_mean"] = round(float(baseline_sharpe.mean()), 4)
                comparison["baseline_sharpe_max"] = round(float(baseline_sharpe.max()), 4)
                causal_sharpe = causal_metrics.get("sharpe", 0)
                comparison["sharpe_percentile"] = round(
                    float((baseline_sharpe < causal_sharpe).mean()), 4,
                )
        except Exception as e:
            comparison["error"] = str(e)

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_backtest(
    factor_path: str,
    factor_column: str,
    data_root: str | None = None,
    output_dir: str = "",
    timespan: str | None = None,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    n_long: int = DEFAULT_LONGX,
    cost_bps: float = DEFAULT_COST_BPS,
    baseline_ic_path: str | None = None,
    use_pandadata: bool = False,
    pandadata_start: str = "",
    pandadata_end: str = "",
    pandadata_indicator: str = "000300",
) -> dict[str, Any]:
    """Run complete standalone backtest on a causal alpha factor."""
    if not HAS_PANDAS:
        return {"error": "pandas_required"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load factor ---
    print("Loading factor data...", file=sys.stderr)
    factor_df = pd.read_csv(factor_path)
    factor_df["date"] = pd.to_datetime(factor_df["date"].astype(str), format="%Y%m%d")
    factor_pivot = factor_df.pivot_table(
        index="date", columns="ticker", values=factor_column, aggfunc="first",
    )
    factor_pivot = factor_pivot.sort_index()
    print(
        f"  Factor: {factor_pivot.shape[0]} dates × "
        f"{factor_pivot.shape[1]} tickers",
        file=sys.stderr,
    )

    if timespan:
        parts = timespan.split()
        if len(parts) >= 1:
            start = pd.to_datetime(parts[0], format="%Y%m%d")
            factor_pivot = factor_pivot[factor_pivot.index >= start]
        if len(parts) >= 2:
            end = pd.to_datetime(parts[1], format="%Y%m%d")
            factor_pivot = factor_pivot[factor_pivot.index <= end]

    # --- Load market data ---
    print("Loading market data...", file=sys.stderr)
    if use_pandadata:
        fields = load_market_data_from_pandadata(
            start_date=pandadata_start,
            end_date=pandadata_end,
            indicator=pandadata_indicator,
        )
    elif data_root:
        fields = load_market_data(data_root)
    else:
        print(
            "[FATAL] Either --data-root or --use-pandadata must be specified.",
            file=sys.stderr,
        )
        return {"error": "no_data_source"}

    if "close" not in fields:
        print(
            "[FATAL] No close price data found. "
            "Ensure trade_price.parquet exists in the data root.",
            file=sys.stderr,
        )
        return {"error": "no_close_price_data"}
    prices = fields["close"]
    print(
        f"  Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers",
        file=sys.stderr,
    )

    # --- Compute forward returns ---
    print("Computing forward returns...", file=sys.stderr)
    fwd_rets = compute_forward_returns(prices, horizon_days=[1, 5, 10, 20])

    # --- Compute cross-sectional rank IC ---
    print("Computing cross-sectional rank IC...", file=sys.stderr)
    ic_results: dict[int, pd.Series] = {}
    for horizon, fwd in fwd_rets.items():
        ic = compute_rank_ic(factor_pivot, fwd)
        if len(ic) > 0:
            ic_results[horizon] = ic
            ic_mean = ic.mean()
            ic_std = ic.std()
            print(
                f"  IC({horizon}d): mean={ic_mean:.4f}, "
                f"std={ic_std:.4f}, "
                f"IR={ic_mean / ic_std.clip(min=1e-8):.2f}, "
                f"n={len(ic)}",
                file=sys.stderr,
            )

    primary_ic = ic_results.get(5, pd.Series(dtype=float))

    # --- Save ICs.csv ---
    if ic_results:
        ic_df = pd.DataFrame({
            f"IC_{h}d": ic_results[h] for h in sorted(ic_results.keys())
        })
        ic_df.index.name = "date"
        ic_df.to_csv(out / "ICs.csv")
        print(f"  Saved: {out / 'ICs.csv'}", file=sys.stderr)

    # IC summary by horizon
    ic_summary_rows = []
    for horizon in sorted(ic_results.keys()):
        ic_s = ic_results[horizon]
        ic_mean = float(ic_s.mean())
        ic_std = float(ic_s.std())
        ic_summary_rows.append({
            "horizon_days": horizon,
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "icir": round(ic_mean / max(ic_std, 1e-8), 4),
            "ic_positive_ratio": round(float((ic_s > 0).mean()), 4),
            "n_obs": len(ic_s),
        })
    ic_summary_df = pd.DataFrame(ic_summary_rows)
    ic_summary_df.to_csv(out / "ic_summary.csv", index=False)
    print(f"  Saved: {out / 'ic_summary.csv'}", file=sys.stderr)

    # --- Simulate portfolio ---
    print(
        f"Simulating portfolio (n_long={n_long}, "
        f"holding={holding_days}d)...",
        file=sys.stderr,
    )
    portfolio_df = simulate_long_only_portfolio(
        factor_pivot, prices,
        holding_days=holding_days,
        n_long=n_long,
        cost_bps=cost_bps,
    )

    if portfolio_df is not None and len(portfolio_df) > 0:
        portfolio_df.to_csv(out / "stats.csv", index=False)
        print(
            f"  Saved: {out / 'stats.csv'} ({len(portfolio_df)} days)",
            file=sys.stderr,
        )

    # --- Compute metrics ---
    metrics = compute_metrics(portfolio_df, primary_ic)

    # --- Compare to baseline ---
    comparison = compare_to_baseline(metrics, baseline_ic_path)

    # --- Build summary ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "factor_path": str(factor_path),
        "factor_column": factor_column,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "config": {
            "holding_days": holding_days,
            "n_long": n_long,
            "cost_bps": cost_bps,
        },
        "performance": metrics,
        "ic_by_horizon": ic_summary_rows,
        "baseline_comparison": comparison,
    }

    summary_path = out / "causal_backtest_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone backtest for causal alpha factors.",
    )
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor-column", required=True)
    parser.add_argument(
        "--data-root", default=None,
        help="Path to market data root (Parquet files). "
             "Required unless --use-pandadata is set.",
    )
    parser.add_argument(
        "--use-pandadata", action="store_true",
        help="Fetch market data from PandaData API instead of local Parquet.",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date YYYYMMDD (required with --use-pandadata).",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date YYYYMMDD (required with --use-pandadata).",
    )
    parser.add_argument(
        "--indicator", default="000300",
        help="Stock universe indicator for PandaData (default: 000300=CSI 300).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timespan", default=None)
    parser.add_argument(
        "--holding-days", type=int, default=DEFAULT_HOLDING_DAYS,
        help=f"Holding days between rebalances (default: {DEFAULT_HOLDING_DAYS}).",
    )
    parser.add_argument(
        "--n-long", type=int, default=DEFAULT_LONGX,
        help=f"Number of long holdings (default: {DEFAULT_LONGX}).",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=DEFAULT_COST_BPS,
        help=f"Transaction cost in bps (default: {DEFAULT_COST_BPS}).",
    )
    parser.add_argument(
        "--baseline-ic", default=None,
        help="Path to baseline performance summary for IC comparison.",
    )
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("[FATAL] pandas is required.", file=sys.stderr)
        sys.exit(1)

    factor_path = Path(args.factor)
    if not factor_path.exists():
        print(f"[FATAL] Factor file not found: {factor_path}", file=sys.stderr)
        sys.exit(1)

    # Validate data source
    if args.use_pandadata:
        if not args.start_date or not args.end_date:
            print(
                "[FATAL] --start-date and --end-date are required "
                "with --use-pandadata.",
                file=sys.stderr,
            )
            sys.exit(1)
        data_root_str = None
    elif args.data_root:
        data_root = Path(args.data_root)
        if not data_root.is_dir():
            print(f"[FATAL] Data root not found: {data_root}", file=sys.stderr)
            sys.exit(1)
        data_root_str = str(data_root)
    else:
        print(
            "[FATAL] Either --data-root or --use-pandadata must be specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60, file=sys.stderr)
    print("CAUSAL ALPHA BACKTEST (STANDALONE ENGINE)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Factor:   {factor_path}", file=sys.stderr)
    print(f"  Column:   {args.factor_column}", file=sys.stderr)
    if args.use_pandadata:
        print(
            f"  Data:     PandaData ({args.start_date}–{args.end_date}, "
            f"indicator={args.indicator})",
            file=sys.stderr,
        )
    else:
        print(f"  Data:     {args.data_root}", file=sys.stderr)
    print(f"  Output:   {args.output_dir}", file=sys.stderr)
    print(
        f"  Config:   holding={args.holding_days}d, "
        f"n_long={args.n_long}, cost={args.cost_bps}bps",
        file=sys.stderr,
    )
    print("=" * 60, file=sys.stderr)

    summary = run_full_backtest(
        factor_path=str(factor_path),
        factor_column=args.factor_column,
        data_root=data_root_str,
        output_dir=args.output_dir,
        timespan=args.timespan,
        holding_days=args.holding_days,
        n_long=args.n_long,
        cost_bps=args.cost_bps,
        baseline_ic_path=args.baseline_ic,
        use_pandadata=args.use_pandadata,
        pandadata_start=args.start_date or "",
        pandadata_end=args.end_date or "",
        pandadata_indicator=args.indicator,
    )

    if "error" in summary:
        print(f"\n[FATAL] {summary['error']}", file=sys.stderr)
        sys.exit(1)

    perf = summary.get("performance", {})
    comp = summary.get("baseline_comparison", {})

    print(f"\n{'='*60}", file=sys.stderr)
    print("BACKTEST RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for label, key in [
        ("Rank IC Mean", "ic_mean"),
        ("ICIR", "icir"),
        ("Sharpe Ratio", "sharpe"),
        ("Max Drawdown", "max_drawdown"),
        ("Daily Win Ratio", "daily_win_ratio"),
    ]:
        if key in perf:
            val = perf[key]
            if key in ("max_drawdown", "daily_win_ratio", "annual_return"):
                print(f"  {label:20s} {val*100:.2f}%", file=sys.stderr)
            else:
                print(f"  {label:20s} {val:.4f}", file=sys.stderr)

    if "annual_return" in perf:
        print(
            f"  {'Annual Return':20s} {perf['annual_return']*100:.2f}%",
            file=sys.stderr,
        )
    if "excess_return" in perf:
        print(
            f"  {'Excess Return':20s} {perf['excess_return']*100:.2f}%",
            file=sys.stderr,
        )

    if "ic_advantage" in comp:
        adv = comp["ic_advantage"]
        direction = "higher" if adv > 0 else "lower"
        print(
            f"\n  vs. Baseline:  {abs(adv)*100:.1f}% {direction} IC",
            file=sys.stderr,
        )
    if "sharpe_percentile" in comp:
        print(
            f"  Sharpe Percentile: {comp['sharpe_percentile']*100:.1f}%",
            file=sys.stderr,
        )

    ic_horizons = summary.get("ic_by_horizon", [])
    if ic_horizons:
        print(f"\n  Multi-Horizon IC:", file=sys.stderr)
        for row in ic_horizons:
            print(
                f"    {row['horizon_days']:3d}d: "
                f"IC={row['ic_mean']:.4f}, IR={row['icir']:.2f}, "
                f">0={row['ic_positive_ratio']*100:.1f}%",
                file=sys.stderr,
            )

    print(f"\n  Summary: {args.output_dir}/causal_backtest_summary.json", file=sys.stderr)
    print(f"  ICs:     {args.output_dir}/ICs.csv", file=sys.stderr)
    print(f"  Stats:   {args.output_dir}/stats.csv", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
