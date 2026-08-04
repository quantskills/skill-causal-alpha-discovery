#!/usr/bin/env python3
"""Generate OHLCV-derived feature pool for causal discovery.

Produces 30–50 derived features from raw OHLCV market data using only the
27 allowed functions on the 6 OHLCV fields. Features are organized into
five families: momentum, volatility, volume/flow, price pattern, and
cross-family interactions.

Usage::

    # From local Parquet data
    python scripts/generate_features.py --data-root ../backtest/market_data_2015_2019/ --output features.csv
    python scripts/generate_features.py --data-root ./market_data/ --output features.csv --families momentum,volatility

    # From PandaData API (A-share real-time)
    python scripts/generate_features.py --use-pandadata --start-date 20150101 --end-date 20191231 --indicator 000300 --output features.csv
    python scripts/generate_features.py --use-pandadata --start-date 20200101 --end-date 20251231 --output features.csv --max-features 40
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
_feat_cfg = _config.get("feature_generation", {})

DEFAULT_FAMILIES: list[str] = _feat_cfg.get(
    "families",
    ["momentum", "volatility", "volume_flow", "price_pattern", "cross_family"],
)
MAX_PER_FAMILY: int = _feat_cfg.get("max_features_per_family", 12)
MAX_TOTAL: int = _feat_cfg.get("max_total_features", 60)
MIN_WINDOW: int = _feat_cfg.get("min_rolling_window", 2)
MAX_WINDOW: int = _feat_cfg.get("max_rolling_window", 252)

# Windows used across feature families
_MOMENTUM_WINDOWS = [5, 10, 20, 60]
_VOLATILITY_WINDOWS = [5, 10, 20, 40, 60, 120]
_VOLUME_WINDOWS = [10, 20, 40, 60]
_CORRELATION_WINDOWS = [10, 20, 60]
_DECAY_WINDOWS = [5, 10, 20]


# ═══════════════════════════════════════════════════════════════════════════════
# OHLCV Data Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_ohlcv(data_root: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV fields from Parquet market data root.

    Expects Parquet files: trade_price.parquet, open_price.parquet,
    balance_price.parquet (high/low), volume, amount, etc.

    Returns dict mapping field name → (date × ticker) DataFrame.
    """
    root = Path(data_root)
    fields: dict[str, pd.DataFrame] = {}

    # Map skill field names to parquet file names
    file_map = {
        "close": "trade_price.parquet",
        "open": "open_price.parquet",
        "high": "balance_price.parquet",  # actually high; adjust if needed
        "low": "balance_price.parquet",    # actually low; adjust if needed
    }

    for field, fname in file_map.items():
        fpath = root / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            if "date" in df.columns and "ticker" in df.columns:
                pivot = df.pivot_table(
                    index="date", columns="ticker",
                    values=df.columns.difference(["date", "ticker"])[0],
                    aggfunc="first",
                )
                pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
                pivot = pivot.sort_index()
                fields[field] = pivot

    # Try to load high/low separately
    for field in ["high", "low"]:
        fpath = root / f"{field}_price.parquet"
        if fpath.exists() and field not in fields:
            df = pd.read_parquet(fpath)
            if "date" in df.columns and "ticker" in df.columns:
                val_col = df.columns.difference(["date", "ticker"])[0]
                pivot = df.pivot_table(
                    index="date", columns="ticker",
                    values=val_col, aggfunc="first",
                )
                pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
                pivot = pivot.sort_index()
                fields[field] = pivot

    # Volume and amount may be in separate parquets or combined
    for field, fname in [("volume", "volume.parquet"), ("amount", "amount.parquet")]:
        fpath = root / fname
        if fpath.exists() and field not in fields:
            df = pd.read_parquet(fpath)
            if "date" in df.columns and "ticker" in df.columns:
                val_col = df.columns.difference(["date", "ticker"])[0]
                pivot = df.pivot_table(
                    index="date", columns="ticker",
                    values=val_col, aggfunc="first",
                )
                pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
                pivot = pivot.sort_index()
                fields[field] = pivot

    # If we couldn't find high/low separately, try from balance_price with separate columns
    if "high" not in fields or "low" not in fields:
        bp = root / "balance_price.parquet"
        if bp.exists():
            df = pd.read_parquet(bp)
            # Try to find high/low in columns
            cols = [c for c in df.columns if c not in ("date", "ticker")]
            for c in cols:
                if "high" in c.lower() and "high" not in fields:
                    pivot = df.pivot_table(index="date", columns="ticker", values=c, aggfunc="first")
                    pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
                    fields["high"] = pivot.sort_index()
                if "low" in c.lower() and "low" not in fields:
                    pivot = df.pivot_table(index="date", columns="ticker", values=c, aggfunc="first")
                    pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
                    fields["low"] = pivot.sort_index()

    # Validate
    required = {"open", "high", "low", "close", "volume", "amount"}
    missing = required - set(fields.keys())
    if missing:
        print(
            f"[WARN] Missing fields: {missing}. "
            f"Features requiring these fields will be skipped.",
            file=sys.stderr,
        )

    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Generation Functions (using only the 27 allowed functions)
# ═══════════════════════════════════════════════════════════════════════════════

def _returns(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x.pct_change(n)


def _delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(n)


def _delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x - x.shift(n)


def _ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).mean()


def _ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).std()


def _ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).max()


def _ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).min()


def _ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).sum()


def _ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).rank(pct=True)


def _decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly-decayed weighted moving average."""
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    result = x.rolling(n, min_periods=max(2, n // 4)).apply(
        lambda s: np.dot(s, weights) if len(s) == n else np.nan, raw=True,
    )
    return result


def _correlation(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(2, n // 4)).corr(y)


def _adv(n: int, volume: pd.DataFrame) -> pd.DataFrame:
    return _ts_mean(volume, n)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def _power(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x ** n


def _signed_power(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return np.sign(x) * (x.abs() ** n)


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.clip(lower=1e-8))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Families
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_momentum(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Momentum family: return-based features."""
    close = fields.get("close")
    if close is None:
        return {}

    features: dict[str, pd.DataFrame] = {}
    for w in _MOMENTUM_WINDOWS:
        if w <= MAX_WINDOW:
            features[f"ret_{w}d"] = _returns(close, w)
            features[f"ret_{w}d_rank"] = _rank(_returns(close, w))

    # Risk-adjusted momentum
    for w_ret in [5, 20, 60]:
        for w_vol in [20, 60]:
            if w_ret <= MAX_WINDOW and w_vol <= MAX_WINDOW:
                vol = _ts_std(_returns(close, 1), w_vol)
                ret = _returns(close, w_ret)
                features[f"mom_{w_ret}d_vol_{w_vol}d"] = ret / vol.clip(lower=1e-8)

    # Decay-linear momentum
    for w in [5, 10, 20]:
        if w <= MAX_WINDOW:
            features[f"decay_ret_{w}d"] = _decay_linear(_returns(close, 1), w)

    # Time-series rank momentum
    for w in [20, 60]:
        if w <= MAX_WINDOW:
            features[f"ts_rank_close_{w}d"] = _ts_rank(close, w)

    return features


def _generate_volatility(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Volatility family: return dispersion and range-based."""
    close = fields.get("close")
    high = fields.get("high")
    low = fields.get("low")
    if close is None:
        return {}

    features: dict[str, pd.DataFrame] = {}
    ret_1d = _returns(close, 1)

    for w in _VOLATILITY_WINDOWS:
        if w <= MAX_WINDOW:
            features[f"vol_{w}d"] = _ts_std(ret_1d, w)

    # Volatility of volatility
    for w in [20, 60]:
        if w <= MAX_WINDOW:
            base_vol = _ts_std(ret_1d, w)
            features[f"vol_of_vol_{w}d"] = _ts_std(base_vol, w * 2) if w * 2 <= MAX_WINDOW else base_vol

    # Volatility regime (short-term vs long-term)
    for w_short, w_long in [(5, 20), (20, 60), (20, 120)]:
        if w_long <= MAX_WINDOW:
            short_vol = _ts_std(ret_1d, w_short)
            long_vol = _ts_std(ret_1d, w_long)
            features[f"vol_ratio_{w_short}_{w_long}"] = short_vol / long_vol.clip(lower=1e-8)

    # Range-based volatility (high-low spread)
    if high is not None and low is not None:
        for w in [5, 20, 60]:
            if w <= MAX_WINDOW:
                hl_range = (high - low) / close.clip(lower=1e-8)
                features[f"hl_range_{w}d"] = _ts_mean(hl_range, w)

    # Downside volatility
    for w in [20, 60]:
        if w <= MAX_WINDOW:
            down_ret = ret_1d.clip(upper=0)
            features[f"down_vol_{w}d"] = _ts_std(down_ret, w)

    return features


def _generate_volume_flow(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Volume/flow family: volume, amount, and their dynamics."""
    volume = fields.get("volume")
    amount = fields.get("amount")
    if volume is None:
        return {}

    features: dict[str, pd.DataFrame] = {}

    # Volume relative to moving average
    for w in _VOLUME_WINDOWS:
        if w <= MAX_WINDOW:
            features[f"vol_ma_{w}d"] = volume / _adv(w, volume).clip(lower=1e-8)

    # Volume volatility (coefficient of variation)
    for w in _VOLUME_WINDOWS:
        if w <= MAX_WINDOW:
            features[f"vol_cv_{w}d"] = _ts_std(volume, w) / _ts_mean(volume, w).clip(lower=1e-8)

    # Volume trend
    for w in [5, 10, 20]:
        if w <= MAX_WINDOW:
            features[f"vol_delta_{w}d"] = _delta(volume, w) / _delay(volume, w).clip(lower=1e-8)

    # Amount features (if available)
    if amount is not None:
        for w in _VOLUME_WINDOWS:
            if w <= MAX_WINDOW:
                features[f"amt_cv_{w}d"] = _ts_std(amount, w) / _ts_mean(amount, w).clip(lower=1e-8)
        for w in [5, 10, 20]:
            if w <= MAX_WINDOW:
                features[f"amt_delta_{w}d"] = _delta(amount, w) / _delay(amount, w).clip(lower=1e-8)

    # Decay-linear volume
    for w in [5, 10, 20]:
        if w <= MAX_WINDOW:
            features[f"decay_vol_{w}d"] = _decay_linear(volume, w)

    return features


def _generate_price_pattern(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Price pattern family: intraday position, candle patterns, divergence."""
    close = fields.get("close")
    high = fields.get("high")
    low = fields.get("low")
    open_ = fields.get("open")
    if close is None:
        return {}

    features: dict[str, pd.DataFrame] = {}

    # Intraday position (close location within day's range)
    if high is not None and low is not None:
        hl_range = (high - low).clip(lower=1e-8)
        features["intraday_pos"] = (close - low) / hl_range

    # Price relative to moving average (z-score)
    for w in [5, 10, 20, 60, 120]:
        if w <= MAX_WINDOW:
            ma = _ts_mean(close, w)
            std = _ts_std(close, w)
            features[f"price_z_{w}d"] = (close - ma) / std.clip(lower=1e-8)

    # Distance from N-day high/low
    for w in [20, 60, 120]:
        if w <= MAX_WINDOW:
            h = _ts_max(close, w)
            l = _ts_min(close, w)
            hl_r = (h - l).clip(lower=1e-8)
            features[f"dist_high_{w}d"] = (close - l) / hl_r

    # Candle body / wick ratios
    if open_ is not None and high is not None and low is not None:
        body = (close - open_).abs()
        upper_wick = high - close.clip(lower=open_)
        lower_wick = open_.clip(lower=close) - low
        features["candle_body_ratio"] = body / (high - low).clip(lower=1e-8)
        features["upper_wick_ratio"] = upper_wick / (high - low).clip(lower=1e-8)

    # Consecutive directional moves
    for w in [5, 10, 20]:
        if w <= MAX_WINDOW:
            direction = _sign(_delta(close, 1))
            features[f"dir_drift_{w}d"] = _ts_sum(direction, w)

    return features


def _generate_cross_family(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cross-family interactions: price-volume, volatility-volume, etc."""
    close = fields.get("close")
    volume = fields.get("volume")
    amount = fields.get("amount")
    high = fields.get("high")
    low = fields.get("low")
    if close is None or volume is None:
        return {}

    features: dict[str, pd.DataFrame] = {}

    # Price-volume correlation
    for w in _CORRELATION_WINDOWS:
        if w <= MAX_WINDOW:
            features[f"corr_close_vol_{w}d"] = _correlation(close, volume, w)
            rank_close = _rank(close)
            rank_vol = _rank(volume)
            features[f"corr_rank_cv_{w}d"] = _correlation(rank_close, rank_vol, w)

    # Volume-confirmed momentum
    for w in [5, 10, 20]:
        if w <= MAX_WINDOW:
            ret = _returns(close, w)
            vol_ratio = volume / _adv(w, volume).clip(lower=1e-8)
            features[f"mom_vol_{w}d"] = ret * vol_ratio

    # Amount-confirmed momentum (if amount available)
    if amount is not None:
        for w in [5, 10, 20]:
            if w <= MAX_WINDOW:
                ret = _returns(close, w)
                amt_ratio = amount / _ts_mean(amount, w).clip(lower=1e-8)
                features[f"mom_amt_{w}d"] = ret * amt_ratio

    # Volatility-volume interaction
    for w in [10, 20]:
        if w <= MAX_WINDOW:
            vol = _ts_std(_returns(close, 1), w)
            vol_ratio = volume / _adv(w, volume).clip(lower=1e-8)
            features[f"vol_times_volratio_{w}d"] = vol * vol_ratio

    # Intraday range with volume
    if high is not None and low is not None:
        for w in [5, 10]:
            if w <= MAX_WINDOW:
                hl_range = (high - low) / close.clip(lower=1e-8)
                vol_ratio = volume / _adv(w, volume).clip(lower=1e-8)
                features[f"range_vol_{w}d"] = _ts_mean(hl_range, w) * vol_ratio

    # Turnover (amount / market cap proxy from close * volume... approximate)
    if amount is not None:
        features["turnover_proxy"] = amount / (_ts_mean(amount, 20).clip(lower=1e-8))

    # Skewness of returns
    for w in [20, 60]:
        if w <= MAX_WINDOW:
            ret_1d = _returns(close, 1)
            features[f"skew_{w}d"] = _ts_mean(_power(ret_1d, 3), w) / (
                _power(_ts_std(ret_1d, w), 3).clip(lower=1e-8)
            )

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "momentum": _generate_momentum,
    "volatility": _generate_volatility,
    "volume_flow": _generate_volume_flow,
    "price_pattern": _generate_price_pattern,
    "cross_family": _generate_cross_family,
}


def generate_features(
    fields: dict[str, pd.DataFrame],
    families: list[str] | None = None,
    max_per_family: int = MAX_PER_FAMILY,
    max_total: int = MAX_TOTAL,
) -> pd.DataFrame:
    """Generate OHLCV-derived features for causal discovery.

    Args:
        fields: Dict mapping field name → (date × ticker) DataFrame.
        families: Which feature families to generate. Default: all.
        max_per_family: Max features per family (caps generation).
        max_total: Overall cap on total features.

    Returns:
        Long-format DataFrame with columns: date, ticker, feature_name, value.
    """
    if families is None:
        families = DEFAULT_FAMILIES

    all_features: dict[str, pd.DataFrame] = {}
    skipped = 0

    for family in families:
        if family not in GENERATORS:
            print(f"[WARN] Unknown family: {family}", file=sys.stderr)
            continue

        gen = GENERATORS[family]
        try:
            feats = gen(fields)
        except Exception as e:
            print(f"[WARN] Family '{family}' failed: {e}", file=sys.stderr)
            continue

        # Limit per family
        family_names = sorted(feats.keys())[:max_per_family]
        for name in family_names:
            if name in all_features:
                skipped += 1
                continue
            all_features[name] = feats[name]

        print(f"  {family}: generated {len(family_names)} features", file=sys.stderr)

    # Cap total
    if len(all_features) > max_total:
        kept = sorted(all_features.keys())[:max_total]
        all_features = {k: all_features[k] for k in kept}

    print(
        f"\nTotal: {len(all_features)} features "
        f"({skipped} skipped as duplicates)",
        file=sys.stderr,
    )

    # Convert to long format: date, ticker, feature_name, value
    rows: list[dict[str, Any]] = []
    for feat_name, df in all_features.items():
        # df is date × ticker
        melted = df.reset_index().melt(
            id_vars="date", var_name="ticker", value_name="value",
        )
        melted["feature_name"] = feat_name
        melted = melted.dropna(subset=["value"])
        rows.append(melted)

    if not rows:
        print("[ERROR] No features generated.", file=sys.stderr)
        sys.exit(1)

    result = pd.concat(rows, ignore_index=True)
    result = result[["date", "ticker", "feature_name", "value"]]
    result["date"] = result["date"].dt.strftime("%Y%m%d")
    result = result.sort_values(["date", "ticker", "feature_name"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OHLCV-derived features for causal discovery.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Path to market data root directory (Parquet files). "
             "Required unless --use-pandadata is set.",
    )
    parser.add_argument(
        "--use-pandadata", action="store_true",
        help="Fetch data from PandaData API instead of local Parquet.",
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
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path for feature pool.",
    )
    parser.add_argument(
        "--families", default=None,
        help="Comma-separated feature families (default: all).",
    )
    parser.add_argument(
        "--max-features", type=int, default=MAX_TOTAL,
        help=f"Maximum total features (default: {MAX_TOTAL}).",
    )
    parser.add_argument(
        "--max-per-family", type=int, default=MAX_PER_FAMILY,
        help=f"Maximum features per family (default: {MAX_PER_FAMILY}).",
    )
    args = parser.parse_args()

    # Parse families
    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(",")]

    # Load data: PandaData or local Parquet
    if args.use_pandadata:
        if not args.start_date or not args.end_date:
            print(
                "[FATAL] --start-date and --end-date are required "
                "with --use-pandadata.",
                file=sys.stderr,
            )
            sys.exit(1)

        from pandadata_loader import (
            fetch_ohlcv_from_pandadata,
            pandadata_to_pivoted_fields,
        )

        print(
            f"Fetching data from PandaData: {args.start_date}–{args.end_date}, "
            f"indicator={args.indicator}",
            file=sys.stderr,
        )
        raw_df = fetch_ohlcv_from_pandadata(
            start_date=args.start_date,
            end_date=args.end_date,
            indicator=args.indicator,
        )
        if raw_df is None:
            print("[FATAL] PandaData returned no data.", file=sys.stderr)
            sys.exit(1)

        fields = pandadata_to_pivoted_fields(raw_df)
    elif args.data_root:
        print(f"Loading OHLCV data from: {args.data_root}", file=sys.stderr)
        fields = load_ohlcv(args.data_root)
    else:
        print(
            "[FATAL] Either --data-root or --use-pandadata must be specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not fields:
        print("[FATAL] No OHLCV fields loaded.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded fields: {list(fields.keys())} "
        f"(shapes: {[(k, v.shape) for k, v in fields.items()]})",
        file=sys.stderr,
    )

    # Generate features
    print("Generating features...", file=sys.stderr)
    features_df = generate_features(
        fields,
        families=families,
        max_per_family=args.max_per_family,
        max_total=args.max_features,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)

    n_feats = features_df["feature_name"].nunique()
    n_dates = features_df["date"].nunique()
    n_stocks = features_df["ticker"].nunique()
    print(
        f"\nSaved: {out_path}\n"
        f"  Features: {n_feats}\n"
        f"  Dates: {n_dates}\n"
        f"  Stocks: {n_stocks}\n"
        f"  Rows: {len(features_df):,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
