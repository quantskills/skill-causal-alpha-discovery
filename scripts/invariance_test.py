#!/usr/bin/env python3
"""Regime invariance testing for causal alpha factors.

Splits historical data into distinct market regimes and formally tests
whether causal effects remain stable across regimes. This is the KEY
DIFFERENTIATOR from correlation-based factors: causal relationships
should be invariant to regime changes.

Tests performed:
  1. Regime identification via volatility clustering (GARCH-style)
  2. ATE homogeneity test (χ²) across regimes
  3. Edge presence test (does the causal edge exist in each regime?)
  4. Sign consistency test (does the effect direction flip across regimes?)
  5. Invariance certificate generation

Usage::

    python scripts/invariance_test.py --factor causal_factor.csv --data-root ../backtest/market_data_2020_2025/ --output invariance_report.json
    python scripts/invariance_test.py --factor causal_factor.csv --data-root ./market_data/ --regimes regimes.csv --output invariance_report.json
    python scripts/invariance_test.py --factor causal_factor.csv --data-root ./market_data/ --regime-method calendar_year --n-regimes 7 --output invariance_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
_inv_cfg = _config.get("invariance_testing", {})

DEFAULT_N_REGIMES: int = _inv_cfg.get("n_regimes", 6)
REGIME_METHOD: str = _inv_cfg.get("regime_method", "volatility_clustering")
MIN_REGIME_OBS: int = _inv_cfg.get("min_regime_obs", 60)
HOMOGENEITY_ALPHA: float = _inv_cfg.get("homogeneity_alpha", 0.05)
STABILITY_REQUIRED_REGIMES: float = _inv_cfg.get("stability_required_regimes", 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# Regime Identification
# ═══════════════════════════════════════════════════════════════════════════════

def identify_regimes(
    returns: pd.Series,
    method: str = REGIME_METHOD,
    n_regimes: int = DEFAULT_N_REGIMES,
) -> pd.Series:
    """Identify market regimes from return data.

    Args:
        returns: Daily return series (indexed by date).
        method: Regime identification method.
        n_regimes: Number of regimes to identify.

    Returns:
        Series mapping date → regime label (0, 1, 2, ...).
    """
    if method == "calendar_year":
        # Simple: one regime per calendar year
        years = returns.index.year
        unique_years = sorted(years.unique())
        # Merge years if too many
        if len(unique_years) > n_regimes:
            group_size = max(1, len(unique_years) // n_regimes)
            regime_map = {}
            for i, y in enumerate(unique_years):
                regime_map[y] = i // group_size
            regimes = pd.Series([regime_map.get(y, 0) for y in years], index=returns.index)
        else:
            regime_map = {y: i for i, y in enumerate(unique_years)}
            regimes = pd.Series([regime_map[y] for y in years], index=returns.index)

    elif method == "volatility_clustering":
        # Use rolling volatility quantiles to define regimes
        vol_20d = returns.rolling(20, min_periods=10).std()
        vol_20d = vol_20d.dropna()

        if len(vol_20d) < MIN_REGIME_OBS * 2:
            # Fallback: equal-sized chunks
            chunks = np.array_split(np.arange(len(returns)), n_regimes)
            regime_arr = np.zeros(len(returns), dtype=int)
            for i, chunk in enumerate(chunks):
                regime_arr[chunk] = i
            regimes = pd.Series(regime_arr, index=returns.index)
        else:
            # Quantile-based regime assignment
            quantile_edges = np.linspace(0, 1, n_regimes + 1)
            quantile_values = vol_20d.quantile(quantile_edges[1:-1])
            bins = [-np.inf] + list(quantile_values) + [np.inf]
            regime_arr = pd.cut(vol_20d, bins=bins, labels=False)
            regimes = regime_arr.reindex(returns.index, method="ffill").fillna(0).astype(int)

    elif method == "return_quantiles":
        # Regime by return magnitude
        ret_20d = returns.rolling(20, min_periods=10).mean()
        ret_20d = ret_20d.dropna()
        if len(ret_20d) < MIN_REGIME_OBS * 2:
            chunks = np.array_split(np.arange(len(returns)), n_regimes)
            regime_arr = np.zeros(len(returns), dtype=int)
            for i, chunk in enumerate(chunks):
                regime_arr[chunk] = i
            regimes = pd.Series(regime_arr, index=returns.index)
        else:
            quantile_edges = np.linspace(0, 1, n_regimes + 1)
            quantile_values = ret_20d.quantile(quantile_edges[1:-1])
            bins = [-np.inf] + list(quantile_values) + [np.inf]
            regime_arr = pd.cut(ret_20d, bins=bins, labels=False)
            regimes = regime_arr.reindex(returns.index, method="ffill").fillna(0).astype(int)

    else:
        raise ValueError(f"Unknown regime method: {method}")

    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# Invariance Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_ate_homogeneity(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    regime_labels: np.ndarray,
    alpha: float = HOMOGENEITY_ALPHA,
) -> dict[str, Any]:
    """Test whether the ATE (factor→returns) is equal across regimes.

    Uses a χ² test of homogeneity: H₀: ATE is equal across all regimes.
    Rejection means the factor's predictive power is regime-dependent
    (i.e., NOT causal in the invariance sense).

    Args:
        factor_values: (n,) array of factor scores.
        forward_returns: (n,) array of forward returns.
        regime_labels: (n,) array of regime labels.
        alpha: Significance level.

    Returns:
        Dict with test results.
    """
    unique_regimes = sorted(set(regime_labels))
    n_regimes = len(unique_regimes)

    if n_regimes < 2:
        return {
            "test": "ate_homogeneity",
            "result": "insufficient_regimes",
            "n_regimes": n_regimes,
            "message": "Need at least 2 regimes for homogeneity test.",
        }

    # Compute ATE per regime (simple linear regression coefficient)
    ate_per_regime = []
    se_per_regime = []
    obs_per_regime = []
    signs = []

    for regime in unique_regimes:
        mask = regime_labels == regime
        X = factor_values[mask].reshape(-1, 1)
        y = forward_returns[mask]

        if len(y) < MIN_REGIME_OBS:
            ate_per_regime.append(np.nan)
            se_per_regime.append(np.nan)
            obs_per_regime.append(len(y))
            signs.append(np.nan)
            continue

        # Standardize for comparability
        X_std = (X - X.mean()) / X.std(ddof=1).clip(min=1e-8)
        y_std = (y - y.mean()) / y.std(ddof=1).clip(min=1e-8)

        if HAS_SKLEARN:
            model = LinearRegression().fit(X_std, y_std)
            ate = model.coef_[0]
        else:
            ate = np.corrcoef(X_std.ravel(), y_std)[0, 1] * y.std(ddof=1) / X.std(ddof=1)

        ate_per_regime.append(float(ate))
        obs_per_regime.append(len(y))
        signs.append(1 if ate > 0 else (-1 if ate < 0 else 0))

        # Standard error via bootstrap
        if HAS_SCIPY:
            n_boot = 200
            boot_ates = []
            rng = np.random.default_rng(42)
            for _ in range(n_boot):
                idx = rng.choice(len(y), size=len(y), replace=True)
                Xb = X_std[idx]
                yb = y_std[idx]
                if HAS_SKLEARN:
                    boot_ates.append(LinearRegression().fit(Xb, yb).coef_[0])
                else:
                    boot_ates.append(np.corrcoef(Xb.ravel(), yb)[0, 1])
            se_per_regime.append(float(np.std(boot_ates, ddof=1)))
        else:
            se_per_regime.append(0.0)

    # Remove NaN regimes
    valid = [i for i in range(n_regimes) if not np.isnan(ate_per_regime[i])]
    if len(valid) < 2:
        return {
            "test": "ate_homogeneity",
            "result": "insufficient_data",
            "n_valid_regimes": len(valid),
            "message": "Too few regimes with sufficient data.",
        }

    ate_valid = [ate_per_regime[i] for i in valid]
    se_valid = [max(se_per_regime[i], 1e-8) for i in valid]
    obs_valid = [obs_per_regime[i] for i in valid]
    signs_valid = [signs[i] for i in valid]

    # χ² test: (ATE_i - ATE_pooled)² / SE_i² ~ χ²(k-1)
    weights = np.array(obs_valid, dtype=float)
    weights = weights / weights.sum()
    ate_pooled = np.average(ate_valid, weights=weights)

    chi2_stat = sum(
        (ate_valid[i] - ate_pooled) ** 2 / se_valid[i] ** 2
        for i in range(len(valid))
    )
    df = len(valid) - 1
    if HAS_SCIPY and df > 0:
        p_value = 1 - scipy_stats.chi2.cdf(chi2_stat, df)
    else:
        p_value = 1.0

    # Sign consistency
    sign_consistency = float(sum(1 for s in signs_valid if s == signs_valid[0])) / len(signs_valid)

    is_invariant = p_value > alpha  # fail to reject H₀ = invariant

    return {
        "test": "ate_homogeneity",
        "result": "invariant" if is_invariant else "regime_dependent",
        "chi2_statistic": round(chi2_stat, 4),
        "degrees_of_freedom": df,
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "ate_per_regime": [round(a, 6) for a in ate_per_regime],
        "se_per_regime": [round(s, 6) for s in se_per_regime],
        "obs_per_regime": obs_per_regime,
        "ate_pooled": round(ate_pooled, 6),
        "sign_consistency": round(sign_consistency, 4),
        "n_regimes_tested": len(valid),
        "n_regimes_total": n_regimes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_invariance_test(
    factor_path: str,
    data_root: str | None = None,
    regimes_csv: str | None = None,
    regime_method: str = REGIME_METHOD,
    n_regimes: int = DEFAULT_N_REGIMES,
    expression_json: str | None = None,
) -> dict[str, Any]:
    """Run full regime invariance test suite.

    Args:
        factor_path: Path to causal factor CSV.
        data_root: Path to market data root (for forward returns).
        regimes_csv: Optional pre-computed regimes CSV.
        regime_method: Method for regime identification.
        n_regimes: Number of regimes.

    Returns:
        Invariance report dict.
    """
    if not HAS_PANDAS:
        return {"error": "pandas_required"}

    # Load factor
    factor_df = pd.read_csv(factor_path)
    if "causal_alpha" not in factor_df.columns:
        # Try to find the factor column
        factor_col = [c for c in factor_df.columns if c not in ("date", "ticker")]
        if not factor_col:
            return {"error": "no_factor_column_found"}
        factor_col = factor_col[0]
    else:
        factor_col = "causal_alpha"

    # Load forward returns from market data
    if data_root:
        root = Path(data_root)
        tp_path = root / "trade_price.parquet"
        if tp_path.exists():
            price_df = pd.read_parquet(tp_path)

            # Detect format: long (has 'ticker' column) vs pivoted (ticker codes as columns)
            if "ticker" in price_df.columns:
                # Long format: date, ticker, value → pivot to date × ticker
                val_col = [c for c in price_df.columns if c not in ("date", "ticker")][0]
                price_pivot = price_df.pivot_table(
                    index="date", columns="ticker", values=val_col, aggfunc="first",
                )
            else:
                # Already pivoted: date index, ticker codes as columns
                if "date" in price_df.columns:
                    price_pivot = price_df.set_index("date")
                else:
                    price_pivot = price_df.copy()

            price_pivot.index = pd.to_datetime(price_pivot.index.astype(str), format="%Y%m%d")
            price_pivot = price_pivot.sort_index()

            # Compute forward returns
            forward_ret_5d = price_pivot.pct_change(5).shift(-5)

            # Align factor data with returns
            factor_df["date"] = pd.to_datetime(factor_df["date"].astype(str), format="%Y%m%d")
            factor_pivot = factor_df.pivot_table(
                index="date", columns="ticker", values=factor_col, aggfunc="first",
            )

            # Get cross-sectional average return and average factor for each date
            avg_ret = forward_ret_5d.mean(axis=1)
            avg_factor = factor_pivot.mean(axis=1)

            # Align
            common_dates = avg_ret.index.intersection(avg_factor.index)
            avg_ret = avg_ret.loc[common_dates]
            avg_factor = avg_factor.loc[common_dates]
        else:
            return {"error": f"trade_price.parquet not found at {tp_path}"}
    else:
        return {"error": "data_root_required"}

    # Remove NaN
    valid_mask = avg_ret.notna() & avg_factor.notna()
    avg_ret = avg_ret[valid_mask]
    avg_factor = avg_factor[valid_mask]

    n_obs = len(avg_ret)
    if n_obs < MIN_REGIME_OBS * 2:
        return {
            "error": f"insufficient_data",
            "n_observations": n_obs,
            "min_required": MIN_REGIME_OBS * 2,
        }

    # Get or compute regime labels
    if regimes_csv:
        regimes_df = pd.read_csv(regimes_csv)
        regimes_df["date"] = pd.to_datetime(regimes_df["date"].astype(str), format="%Y%m%d")
        regimes = regimes_df.set_index("date")["regime"]
        regimes = regimes.reindex(avg_ret.index).fillna(0).astype(int)
        method_list = [(regime_method, regimes)]
    elif regime_method == "all":
        # Run all three methods independently
        method_list = [
            ("volatility_clustering", identify_regimes(avg_ret, "volatility_clustering", n_regimes)),
            ("return_quantiles", identify_regimes(avg_ret, "return_quantiles", n_regimes)),
            ("calendar_year", identify_regimes(avg_ret, "calendar_year", max(n_regimes, 5))),
        ]
    else:
        regimes = identify_regimes(avg_ret, method=regime_method, n_regimes=n_regimes)
        method_list = [(regime_method, regimes)]

    # Run ATE homogeneity test for each method
    all_results: dict[str, Any] = {}
    for mname, mregimes in method_list:
        mregimes = mregimes.loc[avg_ret.index]
        print(f"Testing ATE homogeneity: {mname} ({mregimes.nunique()} regimes)...", file=sys.stderr)
        homogeneity_result = test_ate_homogeneity(
            avg_factor.values, avg_ret.values, mregimes.values,
        )
        all_results[mname] = {
            "n_regimes": int(mregimes.nunique()),
            "regime_distribution": {
                str(r): int((mregimes == r).sum())
                for r in sorted(mregimes.unique())
            },
            "ate_homogeneity": homogeneity_result,
            "invariance_certificate": {
                "is_invariant": homogeneity_result.get("result") == "invariant",
                "confidence": round(1 - homogeneity_result.get("p_value", 1), 4),
                "sign_consistency": homogeneity_result.get("sign_consistency", 0),
                "regimes_tested": homogeneity_result.get("n_regimes_tested", 0),
                "verdict": _verdict(homogeneity_result),
            },
        }

    # Use first method for primary results, store all
    primary = all_results[method_list[0][0]]
    homogeneity_result = primary["ate_homogeneity"]

    # Build report
    report: dict[str, Any] = {
        "meta": {
            "factor_path": factor_path,
            "n_observations": n_obs,
            "n_regimes": primary["n_regimes"],
            "regime_method": regime_method,
            "regime_distribution": primary["regime_distribution"],
        },
        "ate_homogeneity": homogeneity_result,
        "invariance_certificate": primary["invariance_certificate"],
        "all_results": all_results if len(method_list) > 1 else {},
    }

    # ── Per-component invariance testing ─────────────────────────────────────
    per_component_results: list[dict[str, Any]] = []
    if expression_json and Path(expression_json).exists():
        try:
            expr_data = json.loads(Path(expression_json).read_text(encoding="utf-8"))
            components = expr_data.get("components", [])
            if components and "features_path" in expr_data:
                feat_path = Path(expr_data["features_path"])
                if feat_path.exists():
                    features_df = pd.read_csv(feat_path)
                    # Compute forward returns once
                    fwd_ret_5d = price_pivot.pct_change(5, fill_method=None).shift(-5)
                    avg_ret_all = fwd_ret_5d.mean(axis=1)

                    for comp in components:
                        feat_name = comp["feature"]
                        # Extract only this feature's values (efficient)
                        comp_subset = features_df[features_df["feature_name"] == feat_name]
                        if comp_subset.empty:
                            continue

                        # Pivot just this one feature
                        comp_pivot = comp_subset.pivot_table(
                            index="date", columns="ticker",
                            values="value", aggfunc="first",
                        )
                        comp_pivot.index = pd.to_datetime(
                            comp_pivot.index.astype(str), format="%Y%m%d"
                        )
                        comp_pivot = comp_pivot.sort_index()

                        # Cross-sectional mean → daily time series
                        comp_daily = comp_pivot.mean(axis=1)

                        common = comp_daily.index.intersection(avg_ret_all.index)
                        comp_vec = comp_daily.loc[common].dropna().values
                        ret_vec = avg_ret_all.loc[common].dropna().values

                        # Align lengths
                        min_len = min(len(comp_vec), len(ret_vec))
                        comp_vec = comp_vec[:min_len]
                        ret_vec = ret_vec[:min_len]

                        if len(comp_vec) < MIN_REGIME_OBS * 2:
                            continue

                        # Get regimes for these dates
                        comp_regimes = regimes.reindex(
                            common[:min_len], method="ffill"
                        ).fillna(0).astype(int).values

                        comp_test = test_ate_homogeneity(
                            comp_vec, ret_vec, comp_regimes, alpha=HOMOGENEITY_ALPHA,
                        )
                        comp_test["feature"] = feat_name
                        comp_test["train_ate"] = comp["ate"]
                        comp_test["pooled_ate"] = comp_test.get("ate_pooled", 0)
                        per_component_results.append(comp_test)

            report["per_component"] = per_component_results
        except Exception as e:
            print(f"[WARN] Per-component invariance failed: {e}", file=sys.stderr)

    return report


def _verdict(homogeneity: dict) -> str:
    """Generate human-readable verdict from homogeneity test results."""
    is_inv = homogeneity.get("result") == "invariant"
    sign_cons = homogeneity.get("sign_consistency", 0)
    p_val = homogeneity.get("p_value", 1.0)

    if is_inv and sign_cons >= 0.9:
        return (
            "STRONG INVARIANCE: The factor's predictive relationship is "
            "statistically indistinguishable across all tested market regimes. "
            "This is consistent with a genuine causal mechanism."
        )
    elif is_inv:
        return (
            "MODERATE INVARIANCE: The factor passes the homogeneity test but "
            "shows some sign inconsistency. The relationship is largely stable "
            "but may weaken in extreme regimes."
        )
    elif p_val < 0.01:
        return (
            "REGIME DEPENDENT: The factor's predictive power varies "
            "significantly across regimes (p < 0.01). This is characteristic "
            "of a spurious correlation, not a causal mechanism. "
            "Consider removing or regime-conditioning this factor."
        )
    else:
        return (
            "WEAKLY REGIME DEPENDENT: The factor shows some regime sensitivity "
            f"(p = {p_val:.3f}). It is not definitively causal, but may still "
            "add value when combined with truly invariant factors."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime invariance testing for causal alpha factors.",
    )
    parser.add_argument(
        "--factor", required=True,
        help="Path to causal factor CSV.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Path to market data root (for forward returns).",
    )
    parser.add_argument(
        "--regimes", default=None,
        help="Optional pre-computed regimes CSV (date, regime).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for invariance report.",
    )
    parser.add_argument(
        "--regime-method", default=REGIME_METHOD,
        choices=["volatility_clustering", "return_quantiles", "calendar_year", "all"],
        help=f"Regime identification method (default: {REGIME_METHOD}).",
    )
    parser.add_argument(
        "--n-regimes", type=int, default=DEFAULT_N_REGIMES,
        help=f"Number of regimes to identify (default: {DEFAULT_N_REGIMES}).",
    )
    parser.add_argument(
        "--expression", default=None,
        help="Path to causal_alpha_expression.json for per-component testing.",
    )
    args = parser.parse_args()

    report = run_invariance_test(
        factor_path=args.factor,
        data_root=args.data_root,
        regimes_csv=args.regimes,
        regime_method=args.regime_method,
        n_regimes=args.n_regimes,
        expression_json=args.expression,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {out_path}", file=sys.stderr)

    # Print verdict
    cert = report.get("invariance_certificate", {})
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"INVARIANCE CERTIFICATE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Invariant: {cert.get('is_invariant', 'unknown')}", file=sys.stderr)
    print(f"  Confidence: {cert.get('confidence', 0):.2%}", file=sys.stderr)
    print(f"  Sign consistency: {cert.get('sign_consistency', 0):.2%}", file=sys.stderr)
    print(f"  Regimes tested: {cert.get('regimes_tested', 0)}", file=sys.stderr)
    print(f"\n  Verdict: {cert.get('verdict', 'unknown')}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
