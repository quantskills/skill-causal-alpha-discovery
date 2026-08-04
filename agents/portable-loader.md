# Portable Loader Prompt

Use this prompt in Claude Code, Hermes, OpenClaw, or any agent runtime that does
not natively discover `SKILL.md` folders.

```text
You have access to a local skill named causal-alpha at:
<<SKILL_ROOT>>

When the user asks to discover causal alpha factors, test whether existing
factors are causal or merely correlational, find regime-invariant alphas,
or build factors that survive market regime changes:

1. Read <<SKILL_ROOT>>/SKILL.md.
2. Read <<SKILL_ROOT>>/references/causal-discovery.md for the algorithm
   details (PC, LiNGAM, NOTEARS, do-calculus, regime invariance).
3. Read <<SKILL_ROOT>>/config.json for current hyperparameters.
4. Ensure OHLCV market data is available in Parquet format. The skill
   requires the standard 6 OHLCV fields (open, high, low, close, volume,
   amount) and uses only the 27 allowed functions.
5. Run Phase 1 — Generate feature pool:
   python <<SKILL_ROOT>>/scripts/generate_features.py --data-root <market_data_parquet> --output output/run_<ts>/features.csv
6. Run Phase 2 — Causal discovery:
   python <<SKILL_ROOT>>/scripts/causal_discovery.py --features output/run_<ts>/features.csv --target forward_return_5d --output output/run_<ts>/causal_graph.json
7. Run Phase 3 — Build causal alpha:
   python <<SKILL_ROOT>>/scripts/build_causal_alpha.py --graph output/run_<ts>/causal_graph.json --features output/run_<ts>/features.csv --output output/run_<ts>/causal_factor.csv --expression-output output/run_<ts>/causal_alpha_expression.json
8. Run Phase 4 — Invariance testing:
   python <<SKILL_ROOT>>/scripts/invariance_test.py --factor output/run_<ts>/causal_factor.csv --data-root <market_data_parquet> --output output/run_<ts>/invariance_report.json
9. Run Phase 5 — Backtest:
   python <<SKILL_ROOT>>/scripts/integrate_backtest.py --factor output/run_<ts>/causal_factor.csv --factor-column causal_alpha --data-root <market_data_parquet> --output-dir output/run_<ts>/
10. Read causal_graph.json, causal_alpha_expression.json, and
    invariance_report.json. Present:
    - Which OHLCV features are causal parents of forward returns
    - The causal alpha expression (valid OHLCV formula)
    - The invariance certificate (is the factor regime-invariant?)
    - Backtest metrics (Sharpe, IC, MDD)
    - Comparison of IC decay vs. baseline correlational factors
```

Runtime placement notes:
- Codex: keep under a Codex skill path, invoke `$causal-alpha`.
- Claude Code: keep under a Claude skill path, invoke `$causal-alpha`.
- Cursor: copy to `.cursor/skills/causal-alpha`, enable `agents/cursor-rule.mdc`.
- Hermes/OpenClaw: mount as local skill root or paste loader prompt with real path.
