#!/usr/bin/env python3
"""Build causal alpha factor expression from discovered causal graph.

Converts a causal DAG into a valid OHLCV-only factor expression. Identifies
direct causal parents of forward returns, estimates Average Treatment Effects
(ATE) via backdoor adjustment, weights components by ATE × stability, and
outputs both the factor expression string and the computed factor matrix.

The output expression uses ONLY the 6 OHLCV fields (open, high, low, close,
volume, amount) and the 27 allowed functions — fully compatible with the
OHLCV factor contract (6 fields, 27 functions).

Usage::

    python scripts/build_causal_alpha.py --graph causal_graph.json --features features.csv --output causal_factor.csv
    python scripts/build_causal_alpha.py --graph causal_graph.json --features features.csv --max-components 3 --output causal_factor.csv
    python scripts/build_causal_alpha.py --graph causal_graph.json --features features.csv --expression-output expression.json --output causal_factor.csv
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
_ac_cfg = _config.get("alpha_construction", {})

MIN_STABILITY: float = _ac_cfg.get("min_stability_for_inclusion", 0.7)
MAX_COMPONENTS: int = _ac_cfg.get("max_components", 5)
WEIGHT_METHOD: str = _ac_cfg.get("weight_method", "ate_times_stability")
CROSS_SECTIONAL_RANK: bool = _ac_cfg.get("cross_sectional_rank", True)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Expression Mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Maps generated feature names back to valid OHLCV expressions using only
# the 27 allowed functions and 6 fields. This is the BRIDGE between the
# causal discovery vocabulary and the factor contract.
FEATURE_TO_EXPRESSION: dict[str, str] = {
    # Momentum family
    "ret_5d": "returns(close, 5)",
    "ret_10d": "returns(close, 10)",
    "ret_20d": "returns(close, 20)",
    "ret_60d": "returns(close, 60)",
    "ret_5d_rank": "rank(returns(close, 5))",
    "ret_10d_rank": "rank(returns(close, 10))",
    "ret_20d_rank": "rank(returns(close, 20))",
    "ret_60d_rank": "rank(returns(close, 60))",
    "ts_rank_close_20d": "ts_rank(close, 20)",
    "ts_rank_close_60d": "ts_rank(close, 60)",
    "decay_ret_5d": "decay_linear(returns(close, 1), 5)",
    "decay_ret_10d": "decay_linear(returns(close, 1), 10)",
    "decay_ret_20d": "decay_linear(returns(close, 1), 20)",

    # Volatility family
    "vol_5d": "ts_std(returns(close, 1), 5)",
    "vol_10d": "ts_std(returns(close, 1), 10)",
    "vol_20d": "ts_std(returns(close, 1), 20)",
    "vol_40d": "ts_std(returns(close, 1), 40)",
    "vol_60d": "ts_std(returns(close, 1), 60)",
    "vol_120d": "ts_std(returns(close, 1), 120)",
    "down_vol_20d": "ts_std(min(returns(close, 1), 0), 20)",
    "down_vol_60d": "ts_std(min(returns(close, 1), 0), 60)",
    "hl_range_5d": "ts_mean((high - low) / max(close, 1e-8), 5)",
    "hl_range_20d": "ts_mean((high - low) / max(close, 1e-8), 20)",
    "hl_range_60d": "ts_mean((high - low) / max(close, 1e-8), 60)",

    # Volatility ratios
    "vol_ratio_5_20": "ts_std(returns(close, 1), 5) / max(ts_std(returns(close, 1), 20), 1e-8)",
    "vol_ratio_20_60": "ts_std(returns(close, 1), 20) / max(ts_std(returns(close, 1), 60), 1e-8)",
    "vol_ratio_20_120": "ts_std(returns(close, 1), 20) / max(ts_std(returns(close, 1), 120), 1e-8)",

    # Volume/flow family
    "vol_ma_10d": "volume / max(adv(10), 1e-8)",
    "vol_ma_20d": "volume / max(adv(20), 1e-8)",
    "vol_ma_40d": "volume / max(adv(40), 1e-8)",
    "vol_ma_60d": "volume / max(adv(60), 1e-8)",
    "vol_cv_10d": "ts_std(volume, 10) / max(ts_mean(volume, 10), 1e-8)",
    "vol_cv_20d": "ts_std(volume, 20) / max(ts_mean(volume, 20), 1e-8)",
    "vol_cv_40d": "ts_std(volume, 40) / max(ts_mean(volume, 40), 1e-8)",
    "vol_cv_60d": "ts_std(volume, 60) / max(ts_mean(volume, 60), 1e-8)",
    "vol_of_vol_20d": "ts_std(ts_std(returns(close, 1), 20), 40)",
    "vol_delta_5d": "delta(volume, 5) / max(delay(volume, 5), 1e-8)",
    "vol_delta_10d": "delta(volume, 10) / max(delay(volume, 10), 1e-8)",
    "vol_delta_20d": "delta(volume, 20) / max(delay(volume, 20), 1e-8)",
    "amt_cv_20d": "ts_std(amount, 20) / max(ts_mean(amount, 20), 1e-8)",
    "amt_cv_40d": "ts_std(amount, 40) / max(ts_mean(amount, 40), 1e-8)",
    "amt_cv_60d": "ts_std(amount, 60) / max(ts_mean(amount, 60), 1e-8)",
    "amt_delta_10d": "delta(amount, 10) / max(delay(amount, 10), 1e-8)",
    "amt_delta_20d": "delta(amount, 20) / max(delay(amount, 20), 1e-8)",
    "decay_vol_5d": "decay_linear(volume, 5)",
    "decay_vol_10d": "decay_linear(volume, 10)",
    "decay_vol_20d": "decay_linear(volume, 20)",

    # Price pattern family
    "intraday_pos": "(close - low) / max(high - low, 1e-8)",
    "price_z_5d": "(close - ts_mean(close, 5)) / max(ts_std(close, 5), 1e-8)",
    "price_z_10d": "(close - ts_mean(close, 10)) / max(ts_std(close, 10), 1e-8)",
    "price_z_20d": "(close - ts_mean(close, 20)) / max(ts_std(close, 20), 1e-8)",
    "price_z_60d": "(close - ts_mean(close, 60)) / max(ts_std(close, 60), 1e-8)",
    "price_z_120d": "(close - ts_mean(close, 120)) / max(ts_std(close, 120), 1e-8)",
    "dist_high_20d": "(close - ts_min(close, 20)) / max(ts_max(close, 20) - ts_min(close, 20), 1e-8)",
    "dist_high_60d": "(close - ts_min(close, 60)) / max(ts_max(close, 60) - ts_min(close, 60), 1e-8)",
    "dist_high_120d": "(close - ts_min(close, 120)) / max(ts_max(close, 120) - ts_min(close, 120), 1e-8)",
    "candle_body_ratio": "abs(close - open) / max(high - low, 1e-8)",
    "upper_wick_ratio": "(high - max(close, open)) / max(high - low, 1e-8)",
    "dir_drift_5d": "ts_sum(sign(delta(close, 1)), 5)",
    "dir_drift_10d": "ts_sum(sign(delta(close, 1)), 10)",
    "dir_drift_20d": "ts_sum(sign(delta(close, 1)), 20)",

    # Cross-family
    "corr_close_vol_10d": "correlation(close, volume, 10)",
    "corr_close_vol_20d": "correlation(close, volume, 20)",
    "corr_close_vol_60d": "correlation(close, volume, 60)",
    "corr_rank_cv_10d": "correlation(rank(close), rank(volume), 10)",
    "corr_rank_cv_20d": "correlation(rank(close), rank(volume), 20)",
    "corr_rank_cv_60d": "correlation(rank(close), rank(volume), 60)",
    "mom_vol_5d": "returns(close, 5) * volume / max(adv(5), 1e-8)",
    "mom_vol_10d": "returns(close, 10) * volume / max(adv(10), 1e-8)",
    "mom_vol_20d": "returns(close, 20) * volume / max(adv(20), 1e-8)",
    "mom_amt_5d": "returns(close, 5) * amount / max(ts_mean(amount, 5), 1e-8)",
    "mom_amt_10d": "returns(close, 10) * amount / max(ts_mean(amount, 10), 1e-8)",
    "mom_amt_20d": "returns(close, 20) * amount / max(ts_mean(amount, 20), 1e-8)",
    "amt_cv_10d": "ts_std(amount, 10) / max(ts_mean(amount, 10), 1e-8)",
    "amt_cv_20d": "ts_std(amount, 20) / max(ts_mean(amount, 20), 1e-8)",
    "amt_cv_40d": "ts_std(amount, 40) / max(ts_mean(amount, 40), 1e-8)",
    "amt_cv_60d": "ts_std(amount, 60) / max(ts_mean(amount, 60), 1e-8)",
    "skew_20d": "ts_mean(power(returns(close, 1), 3), 20) / max(power(ts_std(returns(close, 1), 20), 3), 1e-8)",
    "skew_60d": "ts_mean(power(returns(close, 1), 3), 60) / max(power(ts_std(returns(close, 1), 60), 3), 1e-8)",
    "turnover_proxy": "amount / max(ts_mean(amount, 20), 1e-8)",

    # Risk-adjusted momentum (generic pattern)
    "mom_5d_vol_20d": "returns(close, 5) / max(ts_std(returns(close, 1), 20), 1e-8)",
    "mom_20d_vol_20d": "returns(close, 20) / max(ts_std(returns(close, 1), 20), 1e-8)",
    "mom_20d_vol_60d": "returns(close, 20) / max(ts_std(returns(close, 1), 60), 1e-8)",
    "mom_60d_vol_20d": "returns(close, 60) / max(ts_std(returns(close, 1), 20), 1e-8)",
    "mom_60d_vol_60d": "returns(close, 60) / max(ts_std(returns(close, 1), 60), 1e-8)",
}


def feature_to_expression(feature_name: str) -> str | None:
    """Convert a generated feature name to a valid OHLCV expression.

    Returns None if the feature cannot be mapped to a valid expression.
    """
    # Direct lookup
    if feature_name in FEATURE_TO_EXPRESSION:
        return FEATURE_TO_EXPRESSION[feature_name]

    # Try fuzzy matching for parametrized features
    # e.g., vol_of_vol_20d → not directly mappable, skip
    # e.g., range_vol_5d → not directly mappable, skip
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Causal Alpha Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_causal_alpha(
    causal_graph: dict[str, Any],
    features_df: pd.DataFrame,
    max_components: int = MAX_COMPONENTS,
    min_stability: float = MIN_STABILITY,
    weight_method: str = WEIGHT_METHOD,
    invariance_report: dict | None = None,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Build a causal alpha factor from discovered causal graph.

    Args:
        causal_graph: Causal graph dict from causal_discovery.py.
        features_df: Long-format features DataFrame.
        max_components: Max causal parents to include.
        min_stability: Minimum bootstrap stability for inclusion.
        weight_method: How to weight components.

    Returns:
        expression: Valid OHLCV factor expression string.
        factor_df: (date × ticker) factor matrix.
        meta: Metadata dict with component breakdown.
    """
    target = causal_graph["meta"]["target"]

    # Get causal parents from the graph
    causal_parents = causal_graph.get("causal_parents_of_target", [])

    # Also check edges targeting the target variable
    target_edges = [
        e for e in causal_graph.get("edges", [])
        if e.get("to") == target and e.get("direction") != "undirected"
    ]
    for edge in target_edges:
        feat = edge["from"]
        effect = edge.get("effect", edge.get("ate", 0.0))
        stability = edge.get("stability", 0.0)
        if not any(p["feature"] == feat for p in causal_parents):
            causal_parents.append({
                "feature": feat,
                "ate": effect if effect else 0.0,
                "stability": stability,
            })

    if not causal_parents:
        print(
            "[WARN] No causal parents found for target. "
            "Cannot build causal alpha.",
            file=sys.stderr,
        )
        return "", pd.DataFrame(), {"error": "no_causal_parents"}

    # Filter and rank parents
    for parent in causal_parents:
        if "stability" not in parent:
            parent["stability"] = 1.0  # default if no bootstrap

    # Filter by stability
    stable_parents = [
        p for p in causal_parents
        if p.get("stability", 0) >= min_stability
    ]

    if not stable_parents:
        print(
            f"[WARN] No parents meet stability threshold {min_stability}. "
            f"Using top-{max_components} by |ATE|.",
            file=sys.stderr,
        )
        stable_parents = sorted(
            causal_parents,
            key=lambda p: abs(p.get("ate", 0)),
            reverse=True,
        )

    # Select top components
    selected = stable_parents[:max_components]

    # Convert feature names to valid OHLCV expressions
    components: list[dict[str, Any]] = []
    unmappable: list[str] = []

    for parent in selected:
        feat_name = parent["feature"]
        expr = feature_to_expression(feat_name)
        if expr is None:
            unmappable.append(feat_name)
            continue

        ate = parent.get("ate", 0.0)
        stability = parent.get("stability", 1.0)

        # Determine sign: positive ATE → positive weight (buy high-factor stocks)
        # negative ATE → negative weight (buy low-factor stocks)
        sign = 1 if ate >= 0 else -1

        # Compute weight
        if weight_method == "ate_times_stability":
            weight = abs(ate) * stability
        elif weight_method == "ate_only":
            weight = abs(ate)
        else:  # equal
            weight = 1.0

        components.append({
            "feature": feat_name,
            "expression": expr,
            "ate": round(ate, 6),
            "stability": round(stability, 4),
            "sign": sign,
            "weight": round(weight, 6),
        })

    if unmappable:
        print(
            f"[WARN] {len(unmappable)} features could not be mapped "
            f"to valid OHLCV expressions: {unmappable}",
            file=sys.stderr,
        )

    if not components:
        print(
            "[FATAL] No components could be mapped to valid OHLCV expressions.",
            file=sys.stderr,
        )
        return "", pd.DataFrame(), {"error": "no_mappable_components"}

    # Normalize weights
    total_weight = sum(c["weight"] for c in components)
    for c in components:
        c["normalized_weight"] = round(c["weight"] / total_weight, 6) if total_weight > 0 else 0

    # Build the factor expression
    terms: list[str] = []
    for c in components:
        sign_str = "-1 * " if c["sign"] < 0 else ""
        expr = c["expression"]
        terms.append(f"{sign_str}rank({expr})")

    expression = " + ".join(terms)
    if len(terms) > 1:
        expression = f"({expression})"

    # ── Auto vol-gate from invariance data ────────────────────────────────────
    vol_gated = False
    if invariance_report:
        all_res = invariance_report.get("all_results", {})
        vol_data = all_res.get("volatility_clustering", {})
        if vol_data:
            ate_list = vol_data.get("ate_homogeneity", {}).get("ate_per_regime", [])
            # Find where the ATE sign flips from negative to positive
            if len(ate_list) >= 4:
                valid_ates = [(i, a) for i, a in enumerate(ate_list) if not np.isnan(a)]
                signs = [1 if a > 0 else (-1 if a < 0 else 0) for _, a in valid_ates]
                # Check: first half consistently negative, second half contains positives
                mid = len(valid_ates) // 2
                first_signs = signs[:mid]
                second_signs = signs[mid:]
                first_neg = sum(1 for s in first_signs if s < 0)
                second_pos = sum(1 for s in second_signs if s > 0)

                if first_neg >= len(first_signs) * 0.66 and second_pos >= 1:
                    vol_gated = True
                    expression = (
                        f"({expression}) * "
                        f"(1 - rank(ts_std(returns(close, 1), 20)))"
                    )
                    print(
                        f"[INFO] Auto vol-gate applied: ATE flips sign in "
                        f"high-vol regimes. Expression wrapped with vol filter.",
                        file=sys.stderr,
                    )

    # Compute factor values from features data
    factor_df = _compute_factor_from_features(features_df, components)

    # Apply vol gate to factor values if enabled
    if vol_gated:
        # Compute vol rank from features: rank(ts_std(returns(close,1), 20))
        vol_feat = "vol_20d"  # maps to ts_std(returns(close,1), 20)
        if vol_feat in features_df["feature_name"].values:
            vol_subset = features_df[features_df["feature_name"] == vol_feat]
            vol_pivot = vol_subset.pivot_table(
                index="date", columns="ticker", values="value", aggfunc="first",
            )
            vol_rank = vol_pivot.rank(axis=1, pct=True)
            # Melt back to long format
            vol_long = vol_rank.reset_index().melt(
                id_vars="date", var_name="ticker", value_name="vol_rank",
            )
            # Merge with factor_df and apply gate
            factor_df = factor_df.merge(vol_long, on=["date", "ticker"], how="left")
            factor_df["causal_alpha"] = factor_df["causal_alpha"] * (1 - factor_df["vol_rank"].fillna(0.5))
            factor_df = factor_df.drop(columns=["vol_rank"])
            factor_df = factor_df.dropna(subset=["causal_alpha"])
            print(
                f"[INFO] Vol gate applied to factor matrix: "
                f"{len(factor_df):,} rows after gating.",
                file=sys.stderr,
            )

    meta: dict[str, Any] = {
        "expression": expression,
        "target": target,
        "n_components": len(components),
        "components": components,
        "vol_gated": vol_gated,
        "unmappable_features": unmappable,
        "weight_method": weight_method,
        "features_path": "",  # set by caller in main()
    }

    return expression, factor_df, meta


def _compute_factor_from_features(
    features_df: pd.DataFrame,
    components: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compute factor matrix from feature values and component weights.

    Pivots features to wide format, then computes weighted combination.
    """
    # Pivot features to date × ticker per feature
    pivot = features_df.pivot_table(
        index="date", columns=["ticker", "feature_name"],
        values="value", aggfunc="first",
    )

    factor_matrix = None
    for c in components:
        feat = c["feature"]
        sign = c["sign"]
        nw = c["normalized_weight"]

        if feat not in pivot.columns.get_level_values(1):
            print(f"[WARN] Feature '{feat}' not in features data.", file=sys.stderr)
            continue

        feat_values = pivot.xs(feat, level=1, axis=1)
        component_factor = feat_values * sign * nw

        if factor_matrix is None:
            factor_matrix = component_factor
        else:
            factor_matrix = factor_matrix.add(component_factor, fill_value=0)

    if factor_matrix is None:
        return pd.DataFrame()

    # Cross-sectional rank normalization per date
    if CROSS_SECTIONAL_RANK:
        factor_matrix = factor_matrix.rank(axis=1, pct=True)

    # Convert to long format
    result = factor_matrix.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="causal_alpha",
    )
    result = result.dropna(subset=["causal_alpha"])
    result = result.sort_values(["date", "ticker"])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal alpha factor from discovered causal graph.",
    )
    parser.add_argument(
        "--graph", required=True,
        help="Path to causal_graph.json from causal_discovery.py.",
    )
    parser.add_argument(
        "--features", required=True,
        help="Path to features CSV (long format).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path for causal factor matrix.",
    )
    parser.add_argument(
        "--expression-output", default=None,
        help="Optional output JSON for alpha expression and component breakdown.",
    )
    parser.add_argument(
        "--max-components", type=int, default=MAX_COMPONENTS,
        help=f"Maximum causal parents to include (default: {MAX_COMPONENTS}).",
    )
    parser.add_argument(
        "--min-stability", type=float, default=MIN_STABILITY,
        help=f"Minimum bootstrap stability for inclusion (default: {MIN_STABILITY}).",
    )
    parser.add_argument(
        "--weight-method", default=WEIGHT_METHOD,
        choices=["ate_times_stability", "ate_only", "equal"],
        help=f"Weighting method (default: {WEIGHT_METHOD}).",
    )
    parser.add_argument(
        "--no-rank", action="store_true",
        help="Disable cross-sectional rank normalization.",
    )
    parser.add_argument(
        "--invariance", default=None,
        help="Path to invariance_report.json for auto vol-gating.",
    )
    args = parser.parse_args()

    # Override global config
    global CROSS_SECTIONAL_RANK
    if args.no_rank:
        CROSS_SECTIONAL_RANK = False

    if not HAS_PANDAS:
        print("[FATAL] pandas is required.", file=sys.stderr)
        sys.exit(1)

    # Load causal graph
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"[FATAL] Causal graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    causal_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    print(f"Loaded causal graph: {len(causal_graph.get('nodes', []))} nodes, "
          f"{len(causal_graph.get('edges', []))} edges", file=sys.stderr)

    # Load features
    features_df = pd.read_csv(args.features)
    print(
        f"Loaded features: {features_df['feature_name'].nunique()} features, "
        f"{len(features_df):,} rows",
        file=sys.stderr,
    )

    # Load optional invariance report for auto vol-gating
    inv_report = None
    if args.invariance:
        inv_path = Path(args.invariance)
        if inv_path.exists():
            inv_report = json.loads(inv_path.read_text(encoding="utf-8"))

    # Build causal alpha
    expression, factor_df, meta = build_causal_alpha(
        causal_graph, features_df,
        max_components=args.max_components,
        min_stability=args.min_stability,
        weight_method=args.weight_method,
        invariance_report=inv_report,
    )

    if factor_df.empty:
        print("[FATAL] Could not build causal alpha factor.", file=sys.stderr)
        sys.exit(1)

    # Save factor matrix
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_csv(out_path, index=False)

    print(f"\nSaved factor: {out_path}", file=sys.stderr)
    print(f"  Expression: {expression}", file=sys.stderr)
    print(f"  Components: {meta['n_components']}", file=sys.stderr)
    for c in meta["components"]:
        print(
            f"    {c['feature']}: ATE={c['ate']:.4f}, "
            f"stability={c['stability']:.2f}, "
            f"weight={c['normalized_weight']:.3f}",
            file=sys.stderr,
        )

    # Save expression metadata
    if args.expression_output:
        meta["features_path"] = str(Path(args.features).resolve())
        expr_path = Path(args.expression_output)
        expr_path.parent.mkdir(parents=True, exist_ok=True)
        expr_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Expression metadata: {expr_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
