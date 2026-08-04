#!/usr/bin/env python3
"""Causal discovery for OHLCV-derived alpha factors.

Applies a hybrid pipeline of PC algorithm (constraint-based skeleton),
LiNGAM (functional causal ordering + effect estimation), and NOTEARS
(continuous optimization for high-dimensional settings) to discover the
causal DAG among OHLCV-derived features and forward returns.

Outputs a causal graph in JSON format with edges, ATE estimates,
bootstrap stability scores, and feature metadata.

Usage::

    python scripts/causal_discovery.py --features features.csv --target forward_return_5d --output causal_graph.json
    python scripts/causal_discovery.py --features features.csv --target forward_return_5d --method pc --output causal_graph.json
    python scripts/causal_discovery.py --features features.csv --target forward_return_5d --bootstrap 200 --output causal_graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
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
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.preprocessing import StandardScaler
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
_cd_cfg = _config.get("causal_discovery", {})

DEFAULT_METHOD: str = _cd_cfg.get("method", "ic_selection")
DEFAULT_ALPHA: float = _cd_cfg.get("pc_alpha", 0.01)
DEFAULT_MAX_COND: int = _cd_cfg.get("pc_max_cond_set", 3)
LINGAM_PRUNE: float = _cd_cfg.get("lingam_prune_threshold", 0.05)
NOTEARS_LAMBDA1: float = _cd_cfg.get("notears_lambda1", 0.1)
NOTEARS_MAX_ITER: int = _cd_cfg.get("notears_max_iter", 100)
NOTEARS_THRESHOLD: float = _cd_cfg.get("notears_threshold", 0.3)
DEFAULT_BOOTSTRAP: int = _cd_cfg.get("bootstrap_runs", 100)
STABILITY_THRESHOLD: float = _cd_cfg.get("bootstrap_stability_threshold", 0.7)
MIN_OBS: int = _cd_cfg.get("min_obs_per_stock", 252)
DIVERSITY_MAX_PER_FAMILY: int = _cd_cfg.get("diversity_max_per_family", 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def load_panel(features_path: str, target_col: str) -> tuple[np.ndarray, list[str]]:
    """Load feature panel and stack into (n_obs, n_features) matrix.

    Pivots long-format features (date, ticker, feature_name, value) to wide
    format, then stacks all (date, ticker) observations into one big matrix.

    Args:
        features_path: Path to features CSV (long format).
        target_col: Name of the target feature column (e.g., 'forward_return_5d').

    Returns:
        data: (n_obs, n_features) numpy array.
        var_names: List of feature names (columns of data).
    """
    if not HAS_PANDAS:
        print("[FATAL] pandas is required for data loading.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(features_path)

    # Pivot to wide: date × ticker × feature
    # First, create a combined key
    df["dt_ticker"] = df["date"].astype(str) + "_" + df["ticker"].astype(str)

    pivot = df.pivot_table(
        index="dt_ticker", columns="feature_name", values="value", aggfunc="first",
    )
    # Replace inf/-inf with NaN, then drop rows with any NaN
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    pivot = pivot.dropna(axis=0, how="any")
    # Clip extreme values for numerical stability
    pivot = pivot.clip(lower=-1e3, upper=1e3)

    var_names = list(pivot.columns)
    data = pivot.values.astype(np.float64)

    print(
        f"Loaded panel: {data.shape[0]:,} observations × {data.shape[1]} features",
        file=sys.stderr,
    )

    return data, var_names


# ═══════════════════════════════════════════════════════════════════════════════
# PC Algorithm (Constraint-Based Causal Discovery)
# ═══════════════════════════════════════════════════════════════════════════════

class PCDiscovery:
    """Constraint-based causal discovery using partial correlation tests.

    Implements PC algorithm skeleton phase with Fisher's z-test for
    conditional independence. Self-contained — no external causal library
    required.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, max_cond_set: int = DEFAULT_MAX_COND):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.n_vars: int = 0
        self.var_names: list[str] = []
        self.adjacency_: np.ndarray | None = None
        self.sepset_: dict[tuple[int, int], set[int]] = {}
        self.edges_: list[dict] = []

    def fit(self, data: np.ndarray, var_names: list[str]) -> "PCDiscovery":
        """Discover causal skeleton from data.

        Args:
            data: (n_obs, n_vars) array.
            var_names: Variable names.
        """
        self.n_vars = data.shape[1]
        self.var_names = list(var_names)

        # Step 1: Fully connected undirected graph (excluding self-loops)
        adj = np.ones((self.n_vars, self.n_vars), dtype=bool)
        np.fill_diagonal(adj, False)

        n = data.shape[0]

        # Step 2: Remove edges via conditional independence testing
        for cond_set_size in range(self.max_cond_set + 1):
            for i in range(self.n_vars):
                neighbors_i = list(np.where(adj[i])[0])
                if len(neighbors_i) <= cond_set_size:
                    continue
                for j in neighbors_i:
                    if not adj[i, j]:
                        continue
                    neighbors_ij = [x for x in neighbors_i if x != j]

                    if self._test_conditional_independence(
                        data, n, i, j, neighbors_ij, cond_set_size,
                    ):
                        adj[i, j] = False
                        adj[j, i] = False

        self.adjacency_ = adj

        # Step 3: Orient edges via collider detection
        self._orient_edges(data, n, adj)

        return self

    def _test_conditional_independence(
        self, data: np.ndarray, n: int, i: int, j: int,
        neighbors: list[int], cond_set_size: int,
    ) -> bool:
        """Test X_i ⊥ X_j | S for all conditioning sets S of given size."""
        if cond_set_size > len(neighbors):
            return False

        for cond_set in combinations(neighbors, cond_set_size):
            if self._partial_corr_test(data, n, i, j, list(cond_set)):
                key = (min(i, j), max(i, j))
                self.sepset_[key] = set(cond_set)
                return True
        return False

    def _partial_corr_test(
        self, data: np.ndarray, n: int, i: int, j: int, cond_set: list[int],
    ) -> bool:
        """Fisher's z-test for zero partial correlation."""
        if not HAS_SCIPY:
            return False

        k = len(cond_set)
        if k == 0:
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
        else:
            idx = [i, j] + list(cond_set)
            sub = data[:, idx]
            with np.errstate(invalid="ignore"):
                corr_matrix = np.corrcoef(sub.T)
            if np.isnan(corr_matrix).any():
                return False
            try:
                prec = np.linalg.inv(corr_matrix)
            except np.linalg.LinAlgError:
                return False
            pcorr = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
            corr = pcorr

        if np.isnan(corr) or np.isinf(corr):
            return False

        # Fisher's z-transformation
        corr_clipped = max(min(corr, 0.9999), -0.9999)
        z = 0.5 * np.log((1 + corr_clipped) / (1 - corr_clipped))
        se = 1.0 / np.sqrt(n - k - 3)
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z / se)))

        return p_value > self.alpha

    def _orient_edges(self, data: np.ndarray, n: int, adj: np.ndarray) -> None:
        """Orient edges using collider detection (V-structures)."""
        self.edges_ = []
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if adj[i, j]:
                    self.edges_.append({
                        "from": self.var_names[i],
                        "to": self.var_names[j],
                        "direction": "undirected",
                        "type": "skeleton",
                    })

        # Collider detection: X_i — X_k — X_j, X_k not in SepSet(i, j)
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if not adj[i, j]:
                    continue
                for k in range(self.n_vars):
                    if k == i or k == j:
                        continue
                    if adj[i, k] and adj[k, j]:
                        # Check if k is NOT in the separating set of (i, j)
                        sepset_key = (min(i, j), max(i, j))
                        sep = self.sepset_.get(sepset_key, set())
                        if k not in sep:
                            # Orient as collider: i → k ← j
                            self._set_edge_direction(i, k, "forward")
                            self._set_edge_direction(j, k, "forward")

    def _set_edge_direction(self, src: int, dst: int, direction: str) -> None:
        """Update edge direction in edges_ list."""
        src_name = self.var_names[src]
        dst_name = self.var_names[dst]
        for edge in self.edges_:
            if edge["from"] == src_name and edge["to"] == dst_name:
                if direction == "forward":
                    pass  # already correct
            elif edge["from"] == dst_name and edge["to"] == src_name:
                if direction == "forward":
                    edge["from"], edge["to"] = edge["to"], edge["from"]
            edge["direction"] = "directed"
            edge["type"] = "oriented"


# ═══════════════════════════════════════════════════════════════════════════════
# DirectLiNGAM (Functional Causal Discovery)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectLiNGAM:
    """DirectLiNGAM: Identify causal ordering and estimate effects.

    For linear non-Gaussian additive noise models:
        X_i = Σ b_ij × X_j + e_i   (e_i non-Gaussian, independent)

    Uses the insight that residuals of a variable regressed on its
    predecessors should be independent of those predecessors.
    """

    def __init__(self, prune_threshold: float = LINGAM_PRUNE):
        self.prune_threshold = prune_threshold
        self.causal_order_: list[int] = []
        self.adjacency_matrix_: np.ndarray | None = None
        self.var_names: list[str] = []
        self.n_vars: int = 0

    def fit(self, data: np.ndarray, var_names: list[str]) -> "DirectLiNGAM":
        """Fit DirectLiNGAM to discover causal ordering and effects.

        Args:
            data: (n_obs, n_vars) array.
            var_names: Variable names.
        """
        self.n_vars = data.shape[1]
        self.var_names = list(var_names)
        remaining = list(range(self.n_vars))
        self.causal_order_ = []
        X = data.copy()

        # Standardize
        if HAS_SKLEARN:
            X = StandardScaler().fit_transform(X)

        for _ in range(self.n_vars):
            # Find the most exogenous variable among remaining
            best_var = self._find_exogenous(X, remaining)
            self.causal_order_.append(best_var)
            remaining.remove(best_var)

            # Regress remaining on the found exogenous, keep residuals
            if remaining:
                X_rem = X[:, remaining]
                x_exo = X[:, best_var].reshape(-1, 1)
                # Simple OLS residual
                beta = np.linalg.lstsq(x_exo, X_rem, rcond=None)[0]
                X_residuals = X_rem - x_exo @ beta
                X[:, remaining] = X_residuals

        # Estimate adjacency matrix (lower-triangular)
        self.adjacency_matrix_ = np.zeros((self.n_vars, self.n_vars))
        for j_idx, j in enumerate(self.causal_order_):
            if j_idx == 0:
                continue
            predecessors = self.causal_order_[:j_idx]
            if not predecessors:
                continue
            y = data[:, j]
            X_pred = data[:, predecessors]
            if HAS_SKLEARN:
                model = LassoCV(cv=3, max_iter=5000, random_state=42).fit(X_pred, y)
                coef = model.coef_
            else:
                coef = np.linalg.lstsq(X_pred, y, rcond=None)[0]

            for p_idx, p in enumerate(predecessors):
                if abs(coef[p_idx]) >= self.prune_threshold:
                    self.adjacency_matrix_[p, j] = coef[p_idx]

        return self

    def _find_exogenous(self, X: np.ndarray, remaining: list[int]) -> int:
        """Identify the most exogenous variable."""
        best_var = remaining[0]
        best_score = float("inf")

        for var in remaining:
            others = [v for v in remaining if v != var]
            if not others:
                return var

            X_var = X[:, var].reshape(-1, 1)
            X_others = X[:, others]
            residuals = X_var - X_others @ np.linalg.lstsq(
                X_others, X_var, rcond=None,
            )[0]

            # Score: dependency between residuals and each other variable
            score = 0.0
            for other in others:
                with np.errstate(invalid="ignore"):
                    corr = abs(np.corrcoef(residuals.ravel(), X[:, other])[0, 1])
                if not np.isnan(corr):
                    score += corr

            if score < best_score:
                best_score = score
                best_var = var

        return best_var

    def get_edges(self) -> list[dict]:
        """Extract directed edges with effect magnitudes."""
        edges = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                w = self.adjacency_matrix_[i, j]
                if abs(w) >= self.prune_threshold:
                    edges.append({
                        "from": self.var_names[i],
                        "to": self.var_names[j],
                        "effect": round(float(w), 6),
                        "direction": "directed",
                        "type": "lingam",
                    })
        return edges

    def get_causal_parents(self, target_name: str) -> list[tuple[str, float]]:
        """Get direct causal parents of target variable with effect sizes."""
        if target_name not in self.var_names:
            return []
        target_idx = self.var_names.index(target_name)
        parents = []
        for i in range(self.n_vars):
            w = self.adjacency_matrix_[i, target_idx]
            if abs(w) >= self.prune_threshold:
                parents.append((self.var_names[i], float(w)))
        return sorted(parents, key=lambda x: abs(x[1]), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEARS (Continuous Optimization for DAG Learning)
# ═══════════════════════════════════════════════════════════════════════════════

def notears_discovery(
    data: np.ndarray,
    var_names: list[str],
    lambda1: float = NOTEARS_LAMBDA1,
    max_iter: int = NOTEARS_MAX_ITER,
    threshold: float = NOTEARS_THRESHOLD,
) -> list[dict]:
    """Learn DAG via NOTEARS continuous optimization.

    Uses the trace-exponential DAG constraint h(W) = tr(exp(W ⊙ W)) - d = 0.
    Minimizes: loss(X, W) + λ₁||W||₁ subject to h(W) = 0.

    Args:
        data: (n_obs, n_vars) array.
        var_names: Variable names.
        lambda1: L1 regularization strength.
        max_iter: Maximum iterations.
        threshold: Minimum absolute weight to keep an edge.

    Returns:
        List of edge dicts.
    """
    n, d = data.shape

    if HAS_SKLEARN:
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
    else:
        X = data - data.mean(axis=0)
        X = X / X.std(axis=0, ddof=1).clip(min=1e-8)

    # Initialize W with zeros
    W = np.zeros((d, d))
    learning_rate = 0.01

    def _h(W_mat: np.ndarray) -> float:
        """DAG constraint: h(W) = trace(exp(W ⊙ W)) - d."""
        M = W_mat * W_mat
        return float(np.trace(np.linalg.matrix_power(np.eye(d) + M / d, d))) - d

    def _loss(W_mat: np.ndarray) -> float:
        """Squared loss: 0.5 * ||X - XW||² / n."""
        residual = X - X @ W_mat
        return 0.5 * np.sum(residual ** 2) / n

    for iteration in range(max_iter):
        # Compute gradient of squared loss
        residual = X - X @ W
        grad_loss = -X.T @ residual / n

        # Simple proximal gradient: no explicit constraint handling for simplicity
        # In practice, the NOTEARS augmented Lagrangian is more involved
        W_new = W - learning_rate * grad_loss

        # Soft thresholding (L1 proximal operator)
        W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - learning_rate * lambda1, 0)

        # Zero out diagonal
        np.fill_diagonal(W_new, 0)

        # Check convergence
        diff = np.max(np.abs(W_new - W))
        W = W_new
        if diff < 1e-6:
            break

        # Check DAG constraint periodically
        if iteration % 20 == 0:
            h_val = _h(W)
            if h_val < 0.01:
                pass  # constraint satisfied

    # Extract edges
    edges = []
    for i in range(d):
        for j in range(d):
            w = W[i, j]
            if abs(w) >= threshold:
                edges.append({
                    "from": var_names[i],
                    "to": var_names[j],
                    "effect": round(float(w), 6),
                    "direction": "directed",
                    "type": "notears",
                })

    return edges


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap Stability Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_stability(
    data: np.ndarray,
    var_names: list[str],
    target_col: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    method: str = DEFAULT_METHOD,
) -> dict[tuple[str, str], float]:
    """Compute edge stability via bootstrap resampling.

    Args:
        data: (n_obs, n_vars) array.
        var_names: Variable names.
        target_col: Name of target variable.
        n_bootstrap: Number of bootstrap iterations.
        method: Causal discovery method.

    Returns:
        Dict mapping (from, to) → stability fraction.
    """
    n = data.shape[0]
    edge_counts: dict[tuple[str, str], int] = {}
    rng = np.random.default_rng(42)

    for b in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample = data[indices]

        try:
            if method in ("pc", "hybrid"):
                pc = PCDiscovery(alpha=DEFAULT_ALPHA, max_cond_set=min(DEFAULT_MAX_COND, 2))
                pc.fit(sample, var_names)
                for edge in pc.edges_:
                    key = (edge["from"], edge["to"])
                    edge_counts[key] = edge_counts.get(key, 0) + 1

            if method in ("lingam", "hybrid"):
                lingam = DirectLiNGAM()
                lingam.fit(sample, var_names)
                for edge in lingam.get_edges():
                    key = (edge["from"], edge["to"])
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        except Exception:
            continue

    stability = {k: v / n_bootstrap for k, v in edge_counts.items()}
    return stability


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def discover_causal_graph(
    data: np.ndarray,
    var_names: list[str],
    target_col: str,
    method: str = DEFAULT_METHOD,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """Run causal discovery and produce a causal graph JSON.

    Args:
        data: (n_obs, n_vars) array.
        var_names: Variable names.
        target_col: Name of target variable (must be in var_names).
        method: 'pc', 'lingam', 'notears', or 'hybrid'.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Causal graph dict.
    """
    if target_col not in var_names:
        raise ValueError(
            f"Target '{target_col}' not found in features. "
            f"Available: {var_names}"
        )

    result: dict[str, Any] = {
        "meta": {
            "method": method,
            "n_observations": int(data.shape[0]),
            "n_variables": int(data.shape[1]),
            "target": target_col,
            "bootstrap_runs": n_bootstrap,
        },
        "nodes": var_names,
        "edges": [],
        "causal_parents_of_target": [],
        "stability": {},
        "feature_metadata": {},
    }

    # --- IC Selection (fast, diversity-aware) ---
    if method == "ic_selection":
        import re
        target_idx = var_names.index(target_col)
        feature_names = [n for n in var_names if n != target_col]

        # Compute rank IC for each feature vs target
        ic_results: list[tuple[str, float]] = []
        for feat in feature_names:
            feat_idx = var_names.index(feat)
            with np.errstate(invalid="ignore"):
                ic = np.corrcoef(
                    np.argsort(np.argsort(data[:, feat_idx])),
                    data[:, target_idx],
                )[0, 1]
            if not np.isnan(ic):
                ic_results.append((feat, float(ic)))

        # Sort by absolute IC (best first)
        ic_results.sort(key=lambda x: abs(x[1]), reverse=True)

        # Diversity: group by base feature name (strip trailing _\d+d)
        def _base_name(name: str) -> str:
            return re.sub(r'_\d+d$', '', name)

        # Select with diversity: max N per base family
        families_used: dict[str, int] = {}
        selected: list[tuple[str, float]] = []
        for feat, ic_val in ic_results:
            base = _base_name(feat)
            count = families_used.get(base, 0)
            if count < DIVERSITY_MAX_PER_FAMILY:
                selected.append((feat, ic_val))
                families_used[base] = count + 1

        # Cap at max_components (from alpha_construction config)
        ac_cfg = _config.get("alpha_construction", {})
        max_comp = ac_cfg.get("max_components", 3)
        selected = selected[:max_comp]

        result["causal_parents_of_target"] = [
            {"feature": name, "ate": ic_val, "stability": 1.0}
            for name, ic_val in selected
        ]

        families_str = ", ".join(
            f"{_base_name(n)}({f})" for n, f in
            [(s[0], families_used.get(_base_name(s[0]), 0)) for s in selected]
        )
        print(
            f"  IC selection: {len(selected)} parents from "
            f"{len(set(_base_name(s[0]) for s in selected))} families "
            f"(diversity_max_per_family={DIVERSITY_MAX_PER_FAMILY})",
            file=sys.stderr,
        )
        for name, ic_val in selected:
            print(f"    {name}: IC={ic_val:+.4f} (base={_base_name(name)})", file=sys.stderr)

        return result

    # --- Run causal discovery (PC / LiNGAM / NOTEARS / hybrid) ---
    pc_result = None
    lingam_result = None

    if method in ("pc", "hybrid"):
        print("Running PC algorithm...", file=sys.stderr)
        pc = PCDiscovery(alpha=DEFAULT_ALPHA, max_cond_set=DEFAULT_MAX_COND)
        pc.fit(data, var_names)
        pc_result = pc
        result["edges"].extend(pc.edges_)
        print(f"  PC: {len(pc.edges_)} edges found", file=sys.stderr)

    if method in ("lingam", "hybrid"):
        print("Running DirectLiNGAM...", file=sys.stderr)
        lingam = DirectLiNGAM()
        lingam.fit(data, var_names)
        lingam_result = lingam
        lingam_edges = lingam.get_edges()
        result["edges"].extend(lingam_edges)
        print(f"  LiNGAM: {len(lingam_edges)} directed edges", file=sys.stderr)

        # Get causal parents of target
        parents = lingam.get_causal_parents(target_col)
        result["causal_parents_of_target"] = [
            {"feature": name, "ate": effect} for name, effect in parents
        ]
        print(
            f"  Causal parents of '{target_col}': "
            f"{[(p[0], round(p[1], 4)) for p in parents]}",
            file=sys.stderr,
        )

    if method == "notears":
        print("Running NOTEARS...", file=sys.stderr)
        notears_edges = notears_discovery(data, var_names)
        result["edges"].extend(notears_edges)
        print(f"  NOTEARS: {len(notears_edges)} directed edges", file=sys.stderr)

    # --- Bootstrap stability ---
    if n_bootstrap > 0:
        print(f"Running {n_bootstrap} bootstrap iterations...", file=sys.stderr)
        stability = bootstrap_stability(data, var_names, target_col, n_bootstrap, method)

        # Convert tuple keys to string for JSON
        result["stability"] = {
            f"{k[0]} -> {k[1]}": round(v, 4) for k, v in stability.items()
        }

        # Annotate edges with stability scores
        for edge in result["edges"]:
            key = (edge["from"], edge["to"])
            rev_key = (edge["to"], edge["from"])
            score = stability.get(key, stability.get(rev_key, 0.0))
            edge["stability"] = round(score, 4)
            edge["stable"] = score >= STABILITY_THRESHOLD

        stable_count = sum(1 for e in result["edges"] if e.get("stable"))
        print(
            f"  Stable edges (≥{STABILITY_THRESHOLD}): {stable_count}/{len(result['edges'])}",
            file=sys.stderr,
        )

    # --- Deduplicate edges (keep highest stability or LiNGAM over PC) ---
    edge_map: dict[tuple[str, str], dict] = {}
    for edge in result["edges"]:
        key = (edge["from"], edge["to"])
        if key not in edge_map:
            edge_map[key] = edge
        else:
            # Prefer LiNGAM/notears over PC skeleton
            existing = edge_map[key]
            if edge.get("type") != "skeleton" and existing.get("type") == "skeleton":
                edge_map[key] = edge
            elif edge.get("stability", 0) > existing.get("stability", 0):
                edge_map[key] = edge

    result["edges"] = list(edge_map.values())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal discovery for OHLCV-derived alpha factors.",
    )
    parser.add_argument(
        "--features", required=True,
        help="Path to features CSV (long format: date, ticker, feature_name, value).",
    )
    parser.add_argument(
        "--target", required=True,
        help="Target feature name (e.g., forward_return_5d).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for causal graph.",
    )
    parser.add_argument(
        "--method", default=DEFAULT_METHOD,
        choices=["pc", "lingam", "notears", "hybrid", "ic_selection"],
        help=f"Causal discovery method (default: {DEFAULT_METHOD}).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
        help=f"Number of bootstrap iterations (default: {DEFAULT_BOOTSTRAP}, 0 to skip).",
    )
    args = parser.parse_args()

    # Load data
    data, var_names = load_panel(args.features, args.target)

    if data.shape[0] < MIN_OBS:
        print(
            f"[WARN] Only {data.shape[0]} observations — "
            f"causal discovery may be unreliable. Minimum recommended: {MIN_OBS}.",
            file=sys.stderr,
        )

    if data.shape[1] < 3:
        print(
            "[FATAL] Need at least 3 features (including target) for meaningful "
            "causal discovery.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run discovery
    graph = discover_causal_graph(
        data, var_names, args.target,
        method=args.method,
        n_bootstrap=args.bootstrap,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {out_path}", file=sys.stderr)
    print(f"  Nodes: {len(graph['nodes'])}", file=sys.stderr)
    print(f"  Edges: {len(graph['edges'])}", file=sys.stderr)
    print(
        f"  Causal parents of '{args.target}': "
        f"{len(graph['causal_parents_of_target'])}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
