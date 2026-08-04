#!/usr/bin/env python3
"""Shared PandaData loader for skill-causal-alpha.

Provides lazy-initialized PandaData API access for fetching A-share daily
OHLCV data. Credentials are read from a ``.env`` file in the skill root.

Usage::

    from pandadata_loader import fetch_ohlcv_from_pandadata

    df = fetch_ohlcv_from_pandadata(
        start_date="20150101",
        end_date="20191231",
        indicator="000300",
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Load .env from skill root
# ═══════════════════════════════════════════════════════════════════════════════

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _SKILL_ROOT / ".env"

if _ENV_PATH.is_file():
    with open(_ENV_PATH, "r", encoding="utf-8") as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key not in os.environ:
                    os.environ[_key] = _val


# ═══════════════════════════════════════════════════════════════════════════════
# PandaData Client
# ═══════════════════════════════════════════════════════════════════════════════

_panda_data = None
_panda_init_attempted = False


def _init_pandadata() -> None:
    """Lazily initialize PandaData. Called only when actually fetching data."""
    global _panda_data, _panda_init_attempted
    if _panda_init_attempted:
        return
    _panda_init_attempted = True

    try:
        import panda_data  # noqa: F811
    except ImportError:
        print(
            "[FATAL] panda_data is not installed. Install it with:\n"
            "    pip install panda_data\n"
            "This skill uses PandaData for real A-share market data.\n"
            "Alternatively, use --data-root to supply local Parquet files.",
            file=sys.stderr,
        )
        sys.exit(1)

    _PANDA_USERNAME = os.environ.get("PANDA_AI_USERNAME", "")
    _PANDA_PASSWORD = os.environ.get("PANDA_AI_PASSWORD", "")
    _PANDA_BASE_URL = os.environ.get(
        "PANDA_AI_BASE_URL", "http://pandadata.pandaaiquant.com",
    )

    if not _PANDA_USERNAME or not _PANDA_PASSWORD:
        print(
            "[FATAL] PandaData credentials not found in .env.\n"
            "Set PANDA_AI_USERNAME and PANDA_AI_PASSWORD in the .env file\n"
            "at the skill root, or use --data-root for local Parquet data.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        panda_data.init_token(
            username=_PANDA_USERNAME,
            password=_PANDA_PASSWORD,
            base_url=_PANDA_BASE_URL,
        )
        _panda_data = panda_data
        print("[INFO] PandaData initialized successfully.", file=sys.stderr)
    except Exception as e:
        print(f"[FATAL] PandaData init failed: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_ohlcv_from_pandadata(
    start_date: str,
    end_date: str,
    indicator: str = "000300",
    exclude_st: bool = True,
) -> pd.DataFrame | None:
    """Fetch A-share daily OHLCV data from PandaData.

    Args:
        start_date: Start date as YYYYMMDD string.
        end_date: End date as YYYYMMDD string.
        indicator: Stock universe index code.
                   Common: 000300=CSI 300, 000905=CSI 500,
                   000016=SSE 50, 000852=CSI 1000.
        exclude_st: Whether to exclude ST (special treatment) stocks.

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close,
        volume, amount. Returns None on failure.
    """
    if _panda_data is None:
        _init_pandadata()
    if _panda_data is None:
        return None

    try:
        raw = _panda_data.get_market_data(
            start_date=start_date,
            end_date=end_date,
            type="stock",
            indicator=indicator,
            st=not exclude_st,
        )

        if raw is None or raw.empty:
            print(
                f"[WARN] PandaData returned empty data for "
                f"{start_date}–{end_date}, indicator={indicator}",
                file=sys.stderr,
            )
            return None

        df = raw.copy()

        # Normalize column names to lowercase
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("open", "high", "low", "close", "volume"):
                col_map[c] = cl
        if col_map:
            df = df.rename(columns=col_map)

        # Compute amount from close * volume if not present
        if "amount" not in df.columns:
            close_col = "close" if "close" in df.columns else None
            volume_col = "volume" if "volume" in df.columns else None
            if close_col and volume_col:
                df["amount"] = (
                    df[close_col].astype(float) * df[volume_col].astype(float)
                )
            else:
                df["amount"] = 0.0

        # Ensure required columns exist
        required = [
            "date", "symbol", "open", "high", "low", "close", "volume", "amount",
        ]
        for col in required:
            if col not in df.columns:
                if col == "amount":
                    df[col] = 0.0
                else:
                    print(
                        f"[WARN] Missing column in PandaData response: {col}",
                        file=sys.stderr,
                    )
                    return None

        df = df[required].copy()
        df["date"] = df["date"].astype(str)
        df["symbol"] = df["symbol"].astype(str)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        n_dates = df["date"].nunique()
        n_symbols = df["symbol"].nunique()
        print(
            f"[INFO] PandaData: {n_dates} dates × {n_symbols} symbols fetched "
            f"({start_date}–{end_date}, indicator={indicator})",
            file=sys.stderr,
        )

        return df

    except Exception as e:
        print(f"[ERROR] PandaData fetch failed: {e}", file=sys.stderr)
        return None


def pandadata_to_pivoted_fields(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Convert PandaData long-format DataFrame to pivoted field dict.

    Args:
        df: Long-format DataFrame with columns:
            date, symbol, open, high, low, close, volume, amount.

    Returns:
        Dict mapping field name → (date × ticker) DataFrame.
    """
    fields: dict[str, pd.DataFrame] = {}
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            continue
        pivot = df.pivot_table(
            index="date", columns="symbol", values=col, aggfunc="first",
        )
        pivot.index = pd.to_datetime(pivot.index.astype(str), format="%Y%m%d")
        pivot = pivot.sort_index()
        fields[col] = pivot

    return fields
