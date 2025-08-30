# Product Requirements Document (PRD)

## Product: Trading Algo Evaluation Dashboard

### 1. **Overview**
The Trading Algo Evaluation Dashboard is a web-based analytics tool that ingests daily CSV files stored in **Backblaze B2**. Each CSV contains performance statistics for multiple trading algorithms (identified by `symphony_sid`). The dashboard helps evaluate and compare trading algos based on risk, return, and stability across different time horizons and asset classes.

---

### 2. **Goals & Objectives**
- Provide a **leaderboard view** of all algos with key performance indicators.
- Enable **risk vs. return visualization** for identifying best-performing algos.
- Compare **Train vs. Out-of-Sample (OOS)** metrics to detect overfitting.
- Display algo performance across **different time horizons** (1M, 3M, 1Y).
- Support **asset class analysis** and grouping.
- Allow drill-down into a single algo’s detailed metrics.

---

### 3. **Users & Use Cases**
**Target Users:**
- Quant researchers validating new algos.
- Portfolio managers choosing algos for allocation.
- Individual traders comparing strategies before deployment.

**Primary Use Cases:**
1. Rank and compare algos by Sharpe ratio, annualized returns, and drawdown.
2. Spot strategies that are robust (consistent train vs OOS performance).
3. Identify which algos perform best in certain asset classes (SPY, BTC, etc.).
4. Deep-dive into a specific algo’s risk-return characteristics.

---

### 4. **Data Source**
- **Input**: CSV files with ~90 columns of statistics.
- **Storage**: Backblaze B2 bucket.
- **Schema highlights**:
  - `symphony_sid`: Unique algo ID.
  - Performance metrics: Sharpe, Calmar, Alpha, Beta, R².
  - Returns: cumulative, annualized, trailing (1M, 3M, 1Y).
  - Risk: max drawdown, volatility.
  - Train vs OOS splits.
  - Metadata: rebalance frequency, asset_classes, created_at.

---

### 5. **Dashboard Features**

#### **5.1 Overview / Header**
- Dropdown: select algo (`symphony_sid`, `name`).
- KPI cards: Best Sharpe, Best Return, Lowest Drawdown, Algo count.

#### **5.2 Algo Leaderboard**
- Table view with sortable columns:
  - Symphony ID, Name, Asset Class, Rebalance Frequency
  - OOS Sharpe, OOS Annualized Return, OOS Max Drawdown, Cumulative Return

#### **5.3 Risk vs Return**
- Scatter plot:
  - X-axis: OOS Annualized Return
  - Y-axis: OOS Sharpe Ratio
  - Bubble size: Max Drawdown

#### **5.4 Train vs OOS Comparison**
- Bar charts comparing:
  - Train Sharpe vs OOS Sharpe
  - Train Return vs OOS Return

#### **5.5 Horizon Performance**
- Multi-bar charts per algo:
  - Trailing 1M, 3M, 1Y returns
- Grouped by asset class (SPY, BTC, Portfolio).

#### **5.6 Risk Metrics**
- Bar chart: OOS Standard Deviation, OOS Max Drawdown per algo.
- Heatmap: Sharpe, Calmar, Alpha correlations across algos.

#### **5.7 Asset Class Breakdown**
- Pie/Treemap: distribution of algos by asset class.
- Average Sharpe/Return per asset group.

#### **5.8 Algo Detail Page (Drill-down)**
- KPI summary (Sharpe, Return, Drawdown, Beta, Alpha).
- Train vs OOS bar comparison.
- Cumulative return timeline.
- Node composition visualization (`num_node_*` fields).

---

### 7. **Success Metrics**
- Users can compare algos in under **5 seconds** via leaderboard.
- Visualization of **risk vs return** clearly highlights top-performing algos.
- **Train vs OOS gaps** easy to spot.
- Ability to select an algo and see all relevant metrics in one place.

---

### 8. **Future Enhancements**
- Add **live performance updates** from brokerage APIs.
- Support **custom backtest uploads** by users.
- Add **optimization filters** (min Sharpe, max drawdown thresholds).
- Export comparisons to PDF/CSV.

