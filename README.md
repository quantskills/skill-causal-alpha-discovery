# Causal Alpha

**简体中文** | [English](README.en.md)

基于因果发现（PC + LiNGAM + NOTEARS）从 OHLCV 数据中挖掘**具有因果不变性**的
alpha 因子。与传统的相关性因子挖掘不同，因果发现识别的是 OHLCV 衍生量之间
*导致*未来收益的结构性关系——产生的因子在跨市场体制下保持稳定。

本技能仅用于研究流程，不构成投资建议。

## 核心理念

传统因子挖掘回答的问题是：

> "历史上这个 OHLCV 模式出现时，收益如何表现？"

这是**相关性**推理。问题在于：金融市场是非平稳的，任何模式与未来收益
之间的相关性都可能（且确实会）随市场体制而变化。

因果发现回答一个根本不同的问题：

> "这个 OHLCV 衍生量是否通过稳定的结构机制*导致*未来收益？"

如果答案是肯定的，那么这种关系是**不变的**——无论牛市、熊市、震荡市、
高波动市，它都成立。

## 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2.（可选）配置 PandaData —— 用于获取 A 股实时数据
cp .env.example .env
# 编辑 .env，填入 PANDA_AI_USERNAME 和 PANDA_AI_PASSWORD
```

## 快速开始

训练/测试期间通过 `config.json` 中的 `data_split` 配置（默认：训练 2015–2020，测试 2021–2025）。因子复杂度通过 `max_components` 控制（默认 3）。

### 方式一：PandaData（A 股实时数据）

```bash
# Step 1: 生成训练期特征（因 API 5 年限制需拆分）
python scripts/generate_features.py --use-pandadata --start-date 20150101 --end-date 20191231 --indicator 000300 --output output/run_001/features_2015_2019.csv
python scripts/generate_features.py --use-pandadata --start-date 20200101 --end-date 20201231 --indicator 000300 --output output/run_001/features_2020.csv
# 合并后添加 forward_return_5d 目标

# Step 2: 因果发现（默认 rank-IC 选择）
python scripts/causal_discovery.py --features output/run_001/features.csv --target forward_return_5d --method ic_selection --output output/run_001/causal_graph.json

# Step 3: 构建因果 alpha（默认最多 3 个成分）
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features.csv --output output/run_001/causal_factor.csv --expression-output output/run_001/causal_alpha_expression.json --max-components 3

# Step 4: 生成测试期特征 & 计算测试期因子值
python scripts/generate_features.py --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output output/run_001/features_test.csv
python scripts/build_causal_alpha.py --graph output/run_001/causal_graph.json --features output/run_001/features_test.csv --output output/run_001/causal_factor_test.csv --max-components 3

# Step 5: 不变性检验 + 回测（测试期）
python scripts/invariance_test.py --factor output/run_001/causal_factor_test.csv --data-root ../backtest/market_data_2020_2025/ --output output/run_001/invariance_report.json
python scripts/integrate_backtest.py --factor output/run_001/causal_factor_test.csv --factor-column causal_alpha --use-pandadata --start-date 20210101 --end-date 20251231 --indicator 000300 --output-dir output/run_001/

# Step 6: 生成综合分析报告
python scripts/generate_report.py --run-dir output/run_001/ --output output/run_001/causal_analysis.md
```

### 方式二：本地 Parquet 数据

```bash
# Step 1: 从本地 Parquet 数据生成特征池
python scripts/generate_features.py \
  --data-root ../backtest/market_data_2015_2019/ \
  --output output/run_001/features.csv

# Step 2–4 同上 ...

# Step 5: 回测验证（本地数据 + 时间范围过滤）
python scripts/integrate_backtest.py \
  --factor output/run_001/causal_factor.csv \
  --factor-column causal_alpha \
  --data-root ../backtest/market_data_2020_2025/ \
  --timespan 20200101 20251231 \
  --output-dir output/run_001/
```

### 进阶用法

```bash
# 指定不同的股票池
python scripts/generate_features.py \
  --use-pandadata --start-date 20200101 --end-date 20251231 \
  --indicator 000905 \        # CSI 500
  --output output/run_002/features.csv

# 回测时自定义参数
python scripts/integrate_backtest.py \
  --factor output/run_001/causal_factor_test.csv \
  --factor-column causal_alpha \
  --use-pandadata --start-date 20210101 --end-date 20251231 \
  --indicator 000300 \
  --holding-days 10 \          # 每 10 个交易日调仓
  --n-long 100 \               # 持有 100 只股票
  --cost-bps 3 \               # 交易成本 3bps
  --output-dir output/run_001/
```

## 股票池代码（PandaData）

| 代码 | 指数 |
|------|------|
| `000300` | 沪深300 |
| `000905` | 中证500 |
| `000016` | 上证50 |
| `000852` | 中证1000 |

## 输出解读

| 文件 | 内容 |
|------|------|
| `causal_graph.json` | 发现的因果 DAG（边、ATE） |
| `causal_alpha_expression.json` | OHLCV 因子表达式 + 成分分解 |
| `causal_factor.csv` / `causal_factor_test.csv` | 每日因子得分（训练/测试期） |
| `invariance_report.json` | 跨体制不变性检验（含每体制 ATE 和日期范围） |
| `causal_backtest_summary.json` | 回测指标（Sharpe、IC、MDD） |
| `causal_analysis.md` | 综合分析报告（18 节，含 Mermaid 图） |
| `ICs.csv` | 每日多周期 Rank IC 序列 |
| `stats.csv` | 组合模拟日频净值 |
| `ic_summary.csv` | 多周期 IC 汇总（1d/5d/10d/20d） |

## 工作原理

1. **生成特征池**：从 6 个 OHLCV 字段生成 88+ 个衍生特征（5 个家族）
2. **因果发现**：默认 rank-IC 选择 + 多样性约束（每家族最多 1 个），排除远期收益防未来偏差。支持多目标周期（5d/10d/20d）
3. **构建因子**：top-3 因果父节点 → 映射为有效 OHLCV 表达式 → 按 ATE 加权 → 截面 rank 归一化
4. **自动波动率门控**：若不变性检验发现高波动体制下 ATE 反转，自动施加 `* (1 - rank(vol))` 门控
5. **不变性检验**：三种体制同时检验（波动率聚类 / 牛熊 / 年度），产生合并判决
6. **回测验证**：内置引擎计算 rank IC（1d/5d/10d/20d）/ 多头组合 / Sharpe / 最大回撤
7. **分析报告**：~16 节综合报告，含信号家族分类、IC 自相关、体制 ATE 表
