"""Streamlit dashboard for the trading assistant.

Displays live candlestick charts, indicators, positions, P&L,
trade journal, account drawdown, system health, and backtesting.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text

from src.config import MES_SPEC, RISK_DEFAULTS, settings
from src.core.database import get_duckdb_connection, get_session, get_sqlite_engine

ET = pytz.timezone("America/New_York")


# -- Page Config --

st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Data Loading --

@st.cache_resource
def _get_duckdb():
    return get_duckdb_connection()


@st.cache_resource
def _get_sqlite():
    return get_sqlite_engine()


def _load_accounts() -> list[dict]:
    """Load accounts from config file."""
    config_path = Path(settings.trading.account_config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent.parent / config_path
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return [{"account_id": "default", "name": "Default", "equity": 50000.0}]


def load_bars(conn: duckdb.DuckDBPyConnection, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
    """Load recent bars from DuckDB."""
    table = f"bars_{timeframe}"
    try:
        df = conn.execute(
            f"SELECT * FROM {table} WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            [settings.trading.symbol, limit],
        ).fetchdf()
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_indicators(conn: duckdb.DuckDBPyConnection, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
    """Load recent indicator values from DuckDB."""
    try:
        df = conn.execute(
            """SELECT * FROM indicator_cache
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC LIMIT ?""",
            [settings.trading.symbol, timeframe, limit],
        ).fetchdf()
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_trades(engine, limit: int = 50, account_id: str | None = None) -> pd.DataFrame:
    """Load recent trades from SQLite."""
    try:
        with engine.connect() as conn:
            if account_id and account_id != "All Accounts":
                df = pd.read_sql(
                    text("SELECT * FROM trades WHERE account_id = :acct ORDER BY created_at DESC LIMIT :limit"),
                    conn,
                    params={"acct": account_id, "limit": limit},
                )
            else:
                df = pd.read_sql(
                    text("SELECT * FROM trades ORDER BY created_at DESC LIMIT :limit"),
                    conn,
                    params={"limit": limit},
                )
        return df
    except Exception:
        return pd.DataFrame()


def load_daily_summary(engine, account_id: str | None = None) -> pd.DataFrame:
    """Load daily summaries."""
    try:
        with engine.connect() as conn:
            if account_id and account_id != "All Accounts":
                df = pd.read_sql(
                    text("SELECT * FROM daily_summary WHERE account_id = :acct ORDER BY date DESC LIMIT 30"),
                    conn,
                    params={"acct": account_id},
                )
            else:
                df = pd.read_sql(
                    text("SELECT * FROM daily_summary ORDER BY date DESC LIMIT 30"),
                    conn,
                )
        return df
    except Exception:
        return pd.DataFrame()


def load_equity_snapshots(engine, limit: int = 500, account_id: str | None = None) -> pd.DataFrame:
    """Load equity snapshots."""
    try:
        with engine.connect() as conn:
            if account_id and account_id != "All Accounts":
                df = pd.read_sql(
                    text("SELECT * FROM equity_snapshots WHERE account_id = :acct ORDER BY snapshot_time DESC LIMIT :limit"),
                    conn,
                    params={"acct": account_id, "limit": limit},
                )
            else:
                df = pd.read_sql(
                    text("SELECT * FROM equity_snapshots ORDER BY snapshot_time DESC LIMIT :limit"),
                    conn,
                    params={"limit": limit},
                )
        if not df.empty:
            df = df.sort_values("snapshot_time").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_drawdown_data(engine) -> pd.DataFrame:
    """Load drawdown tracking data."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM drawdown_tracking ORDER BY snapshot_time DESC LIMIT 500"),
                conn,
            )
        if not df.empty:
            df = df.sort_values("snapshot_time").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# -- Chart Building --

def build_price_chart(bars_df: pd.DataFrame, indicators_df: pd.DataFrame) -> go.Figure:
    """Build candlestick chart with indicator overlays."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("MES Price", "RSI (14)", "Volume"),
    )

    if bars_df.empty:
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=bars_df["timestamp"],
            open=bars_df["open"],
            high=bars_df["high"],
            low=bars_df["low"],
            close=bars_df["close"],
            name="MES",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Overlay indicators if available
    if not indicators_df.empty and "timestamp" in indicators_df.columns:
        # Bollinger Bands
        if "bb_upper" in indicators_df.columns:
            valid = indicators_df.dropna(subset=["bb_upper"])
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["bb_upper"],
                    name="BB Upper", line=dict(color="rgba(255,165,0,0.4)", width=1),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["bb_middle"],
                    name="BB Mid", line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dash"),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["bb_lower"],
                    name="BB Lower", line=dict(color="rgba(255,165,0,0.4)", width=1),
                    fill="tonexty", fillcolor="rgba(255,165,0,0.05)",
                ), row=1, col=1)

        # VWAP
        if "vwap" in indicators_df.columns:
            valid = indicators_df.dropna(subset=["vwap"])
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["vwap"],
                    name="VWAP", line=dict(color="#2196F3", width=2),
                ), row=1, col=1)

        # EMA 9 & 21
        if "ema_9" in indicators_df.columns:
            valid = indicators_df.dropna(subset=["ema_9"])
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["ema_9"],
                    name="EMA 9", line=dict(color="#4CAF50", width=1),
                ), row=1, col=1)
        if "ema_21" in indicators_df.columns:
            valid = indicators_df.dropna(subset=["ema_21"])
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["ema_21"],
                    name="EMA 21", line=dict(color="#FF9800", width=1),
                ), row=1, col=1)

        # RSI subplot
        if "rsi_14" in indicators_df.columns:
            valid = indicators_df.dropna(subset=["rsi_14"])
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid["timestamp"], y=valid["rsi_14"],
                    name="RSI", line=dict(color="#9C27B0", width=1),
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Volume bars
    if "volume" in bars_df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(bars_df["close"], bars_df["open"])]
        fig.add_trace(
            go.Bar(x=bars_df["timestamp"], y=bars_df["volume"], name="Volume",
                   marker_color=colors, opacity=0.5),
            row=3, col=1,
        )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Vol", row=3, col=1)

    return fig


# -- Trading Window Helper --

def _get_trading_window_status() -> tuple[bool, str]:
    """Check if current time is within the 10AM-2PM ET trading window."""
    now_et = datetime.now(ET)
    start = time(10, 0)
    end = time(14, 0)
    current = now_et.time()
    is_active = start <= current <= end
    if is_active:
        mins_left = (datetime.combine(now_et.date(), end) - datetime.combine(now_et.date(), current)).seconds // 60
        return True, f"Active ({mins_left}m remaining)"
    elif current < start:
        mins_until = (datetime.combine(now_et.date(), start) - datetime.combine(now_et.date(), current)).seconds // 60
        return False, f"Opens in {mins_until}m"
    else:
        return False, "Closed for today"


# -- Backtesting Tab --

def _render_backtest_tab():
    """Render the backtesting controls and results."""
    st.subheader("Backtest Runner")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_bars = st.number_input("Synthetic bars", min_value=200, max_value=10000, value=2000, step=100)
    with col2:
        bt_equity = st.number_input("Starting equity ($)", min_value=1000.0, max_value=100000.0, value=50000.0, step=1000.0)
    with col3:
        bt_volatility = st.number_input("Volatility", min_value=0.5, max_value=10.0, value=2.5, step=0.5)

    col_regime, col_run = st.columns([1, 1])
    with col_regime:
        use_regime = st.checkbox("Enable regime detection", value=True)
    with col_run:
        run_backtest = st.button("Run Backtest", type="primary")

    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                from src.core.logging import setup_logging
                setup_logging()

                from src.backtesting.engine import BacktestEngine
                from src.market_data.historical import generate_sample_bars
                from src.strategies.base import StrategyConfig
                from src.strategies.mean_reversion import MeanReversionStrategy

                bars = generate_sample_bars(count=bt_bars, start_price=5950.0, volatility=bt_volatility)
                config = StrategyConfig(name="mean_reversion")
                strategy = MeanReversionStrategy(config=config)
                engine = BacktestEngine(
                    strategies=[strategy],
                    starting_equity=bt_equity,
                    use_regime_detection=use_regime,
                )
                results = engine.run(bars)

                st.success("Backtest complete!")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Starting Equity", f"${results.starting_equity:,.2f}")
                mcol2.metric("Ending Equity", f"${results.ending_equity:,.2f}")
                pnl = results.ending_equity - results.starting_equity
                mcol3.metric("Net P&L", f"${pnl:+,.2f}")
                mcol4.metric("Total Trades", len(results.trades))

                if results.metrics:
                    m = results.metrics
                    mcol5, mcol6, mcol7, mcol8 = st.columns(4)
                    mcol5.metric("Win Rate", f"{m.win_rate:.1%}")
                    mcol6.metric("Profit Factor", f"{m.profit_factor:.2f}" if m.profit_factor else "N/A")
                    mcol7.metric("Max Drawdown", f"${m.max_drawdown:.2f}" if m.max_drawdown else "N/A")
                    mcol8.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio else "N/A")

                if results.trades:
                    equity_values = [bt_equity]
                    for t in results.trades:
                        equity_values.append(equity_values[-1] + t.pnl_dollars)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=equity_values, mode="lines", name="Equity",
                        line=dict(color="#2196F3", width=2),
                    ))
                    fig.update_layout(
                        title="Backtest Equity Curve",
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=50, r=20, t=40, b=20),
                        yaxis_title="Equity ($)",
                        xaxis_title="Trade #",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    trade_data = []
                    for t in results.trades:
                        trade_data.append({
                            "Direction": t.direction.value if hasattr(t.direction, "value") else str(t.direction),
                            "Entry": f"{t.entry_price:.2f}",
                            "Exit": f"{t.exit_price:.2f}" if t.exit_price else "-",
                            "P&L": f"${t.pnl_dollars:+.2f}",
                            "Strategy": t.strategy,
                        })
                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
                else:
                    st.info("No trades generated. Try more bars or higher volatility.")

            except Exception as e:
                st.error(f"Backtest error: {e}")

    # Optimizer section
    st.divider()
    st.subheader("Parameter Optimization")

    col1, col2 = st.columns(2)
    with col1:
        opt_trials = st.number_input("Optuna trials", min_value=5, max_value=1000, value=50, step=10)
    with col2:
        opt_bars = st.number_input("Bars for optimization", min_value=500, max_value=10000, value=2000, step=500)

    run_optimizer = st.button("Run Optimization", type="secondary")

    if run_optimizer:
        with st.spinner(f"Running {opt_trials} optimization trials..."):
            try:
                from src.core.logging import setup_logging
                setup_logging()

                from src.backtesting.optimizer import OptunaOptimizer
                from src.market_data.historical import generate_sample_bars

                bars = generate_sample_bars(count=opt_bars, start_price=5950.0, volatility=bt_volatility)
                optimizer = OptunaOptimizer(bars=bars, n_trials=opt_trials, starting_equity=bt_equity)
                best_params, best_score = optimizer.optimize()

                st.success(f"Optimization complete! Best score: {best_score:.4f}")

                param_data = [{"Parameter": k, "Value": f"{v:.4f}"} for k, v in sorted(best_params.items())]
                st.dataframe(pd.DataFrame(param_data), use_container_width=True)

            except Exception as e:
                st.error(f"Optimization error: {e}")


# -- Drawdown Tab --

def _render_drawdown_tab(sqlite, accounts: list[dict], selected_account: str):
    """Render Apex drawdown and profit goal tracking."""
    st.subheader("Apex Account Status")

    for acct in accounts:
        aid = acct["account_id"]
        if selected_account != "All Accounts" and aid != selected_account:
            continue

        equity = acct.get("equity", 50000.0)
        starting_equity = acct.get("equity", 50000.0)  # Base starting equity
        max_eq = acct.get("max_equity_high", equity)
        threshold = acct.get("trailing_drawdown", 2500.0)
        profit_goal = acct.get("profit_goal", 3000.0)
        acct_type = acct.get("account_type", "EVAL")

        # Drawdown calculations
        floor = max_eq - threshold
        remaining = equity - floor
        used_pct = 1.0 - (remaining / threshold) if threshold > 0 else 0.0

        # Profit goal calculations
        realized_profit = equity - starting_equity
        goal_pct = max(0.0, realized_profit / profit_goal) if profit_goal > 0 else 0.0
        profit_remaining = max(0.0, profit_goal - realized_profit)

        st.markdown(f"### {acct.get('name', aid)}")
        st.caption(f"{acct_type} | Max {acct.get('max_contracts', 10)} contracts | Trailing: ${threshold:,.0f} | Goal: ${profit_goal:,.0f}")

        # Profit Goal Progress
        st.markdown("**Eval Profit Goal**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Profit Goal", f"${profit_goal:,.0f}")
        pcol2.metric("Current P&L", f"${realized_profit:+,.2f}")
        pcol3.metric("Remaining", f"${profit_remaining:,.2f}")
        pcol4.metric("Progress", f"{goal_pct:.0%}")

        if goal_pct >= 1.0:
            st.success(f"EVAL PASSED! Profit ${realized_profit:,.2f} >= ${profit_goal:,.0f} goal")
        elif goal_pct >= 0.75:
            st.progress(min(goal_pct, 1.0), text=f"Profit goal: {goal_pct:.0%} - ALMOST THERE!")
        else:
            st.progress(min(goal_pct, 1.0), text=f"Profit goal: {goal_pct:.0%}")

        # Trailing Drawdown
        st.markdown("**Trailing Drawdown**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Equity", f"${equity:,.2f}")
        col2.metric("High Water Mark", f"${max_eq:,.2f}")
        col3.metric("Drawdown Floor", f"${floor:,.2f}")
        col4.metric("Remaining", f"${remaining:,.2f}",
                     delta=f"{(1 - used_pct) * 100:.0f}% available")

        # Drawdown gauge
        if used_pct < 0.5:
            st.progress(used_pct, text=f"Drawdown used: {used_pct:.0%}")
        elif used_pct < 0.8:
            st.progress(used_pct, text=f"Drawdown used: {used_pct:.0%} - CAUTION")
        else:
            st.progress(min(used_pct, 1.0), text=f"Drawdown used: {used_pct:.0%} - DANGER")

        st.divider()

    # Historical drawdown data
    dd_df = load_drawdown_data(sqlite)
    if not dd_df.empty:
        st.divider()
        st.subheader("Drawdown History")
        if selected_account != "All Accounts":
            dd_df = dd_df[dd_df["account_id"] == selected_account]

        if not dd_df.empty:
            fig = go.Figure()
            for aid in dd_df["account_id"].unique():
                acct_df = dd_df[dd_df["account_id"] == aid]
                fig.add_trace(go.Scatter(
                    x=acct_df["snapshot_time"],
                    y=acct_df["drawdown_remaining"],
                    name=aid,
                    mode="lines",
                ))
            fig.update_layout(
                title="Drawdown Remaining Over Time",
                template="plotly_dark",
                height=350,
                yaxis_title="$ Remaining",
            )
            st.plotly_chart(fig, use_container_width=True)


# -- Main App --

def main():
    duck = _get_duckdb()
    sqlite = _get_sqlite()
    accounts = _load_accounts()
    account_ids = ["All Accounts"] + [a["account_id"] for a in accounts]

    # Sidebar
    with st.sidebar:
        st.title("Trading Assistant")
        st.divider()

        # Account selector
        st.subheader("Account")
        selected_account = st.selectbox("Select Account", account_ids, index=0)

        st.divider()

        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mode", "PAPER" if settings.trading.paper_mode else "LIVE")
        with col2:
            st.metric("Symbol", settings.trading.symbol)

        st.metric("Accounts", f"{len(accounts)} loaded")

        # Show selected account equity
        if selected_account != "All Accounts":
            for a in accounts:
                if a["account_id"] == selected_account:
                    st.metric("Account Equity", f"${a.get('equity', 0):,.2f}")
                    st.caption(f"Type: {a.get('account_type', 'EVAL')} | Max: {a.get('max_contracts', 10)} contracts")
                    break

        st.divider()

        # Trading Window status
        st.subheader("Trading Window")
        window_active, window_status = _get_trading_window_status()
        now_et = datetime.now(ET)
        st.caption(f"ET: {now_et.strftime('%I:%M %p')}")
        if window_active:
            st.success(f"10AM-2PM ET: {window_status}")
        else:
            st.warning(f"10AM-2PM ET: {window_status}")

        st.divider()

        # Regime Detection status
        st.subheader("Regime Detection")
        st.caption("ADX + BB/KC squeeze on 5m bars")
        st.info("Awaiting live data")
        st.caption("RANGING = full signals | TRANSITIONAL = 50% | TRENDING = blocked")

        st.divider()

        # Upcoming News Events
        st.subheader("News Blackouts")
        st.caption("FOMC: 30m pre / 60m post")
        st.caption("NFP: 15m pre / 45m post")
        st.caption("CPI: 15m pre / 30m post")
        st.info("No events loaded. Add via NewsCalendar when live.")

        st.divider()

        # Timeframe selector
        timeframe = st.selectbox("Timeframe", ["1m", "5m"], index=0)
        bar_count = st.slider("Bars to display", 50, 500, 200)

        st.divider()

        # Auto-refresh
        refresh = st.checkbox("Auto-refresh (5s)", value=False)
        if refresh:
            st.rerun()

    # Main content
    tab_chart, tab_journal, tab_analytics, tab_drawdown, tab_backtest, tab_health = st.tabs(
        ["Chart", "Trade Journal", "Analytics", "Drawdown", "Backtesting", "System Health"]
    )

    with tab_chart:
        bars_df = load_bars(duck, timeframe, bar_count)
        indicators_df = load_indicators(duck, timeframe, bar_count)

        if bars_df.empty:
            st.info("No market data yet. Connect to Tradovate and start receiving bars.")
        else:
            fig = build_price_chart(bars_df, indicators_df)
            st.plotly_chart(fig, use_container_width=True)

            if not bars_df.empty:
                last = bars_df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Price", f"{last['close']:.2f}")
                col2.metric("High", f"{last['high']:.2f}")
                col3.metric("Low", f"{last['low']:.2f}")
                col4.metric("Volume", f"{last['volume']:,}")

    with tab_journal:
        acct_filter = selected_account if selected_account != "All Accounts" else None
        trades_df = load_trades(sqlite, account_id=acct_filter)
        if trades_df.empty:
            st.info("No trades recorded yet.")
        else:
            closed = trades_df[trades_df["status"] == "CLOSED"]
            if not closed.empty:
                col1, col2, col3, col4 = st.columns(4)
                total = len(closed)
                winners = len(closed[closed["pnl_dollars"] > 0]) if "pnl_dollars" in closed.columns else 0
                win_rate = winners / total * 100 if total > 0 else 0
                total_pnl = closed["pnl_dollars"].sum() if "pnl_dollars" in closed.columns else 0

                col1.metric("Total Trades", total)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Net P&L", f"${total_pnl:.2f}")
                col4.metric("Avg Trade", f"${total_pnl / total:.2f}" if total > 0 else "$0")

            display_cols = [
                c for c in ["account_id", "entry_time", "direction", "entry_price", "exit_price",
                            "pnl_dollars", "strategy", "status", "signal_confidence"]
                if c in trades_df.columns
            ]
            st.dataframe(trades_df[display_cols] if display_cols else trades_df, use_container_width=True)

    with tab_analytics:
        acct_filter = selected_account if selected_account != "All Accounts" else None
        daily_df = load_daily_summary(sqlite, account_id=acct_filter)
        equity_df = load_equity_snapshots(sqlite, account_id=acct_filter)

        if equity_df.empty and daily_df.empty:
            st.info("No analytics data yet. Start paper trading to see performance metrics.")
        else:
            if not equity_df.empty:
                fig = go.Figure()
                if "account_id" in equity_df.columns:
                    for aid in equity_df["account_id"].unique():
                        acct_df = equity_df[equity_df["account_id"] == aid]
                        fig.add_trace(go.Scatter(
                            x=acct_df["snapshot_time"], y=acct_df["equity"],
                            mode="lines", name=aid,
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=equity_df["snapshot_time"], y=equity_df["equity"],
                        mode="lines", name="Equity",
                        line=dict(color="#2196F3", width=2),
                    ))
                fig.update_layout(
                    title="Equity Curve", template="plotly_dark",
                    height=400, margin=dict(l=50, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            if not daily_df.empty:
                st.subheader("Daily Performance")
                st.dataframe(daily_df, use_container_width=True)

    with tab_drawdown:
        _render_drawdown_tab(sqlite, accounts, selected_account)

    with tab_backtest:
        _render_backtest_tab()

    with tab_health:
        st.subheader("Service Health")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tradovate", "Check pending...")
        col2.metric("DuckDB", "Check pending...")
        col3.metric("SQLite", "Check pending...")
        col4.metric("Ollama", "Check pending...")

        st.divider()

        st.subheader("Risk Configuration")
        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            st.markdown("**Position Sizing**")
            st.json({
                "max_risk_per_trade": f"{RISK_DEFAULTS['max_risk_per_trade']:.1%}",
                "max_position_size": RISK_DEFAULTS["max_position_size"],
                "max_concurrent_positions": RISK_DEFAULTS["max_concurrent_positions"],
                "always_use_stop_loss": RISK_DEFAULTS["always_use_stop_loss"],
                "max_stop_distance_atr": f"{RISK_DEFAULTS['max_stop_distance_atr']}x",
            })

        with risk_col2:
            st.markdown("**Loss Limits**")
            st.json({
                "daily_loss_limit": f"{RISK_DEFAULTS['daily_loss_limit']:.0%}",
                "weekly_loss_limit": f"{RISK_DEFAULTS['weekly_loss_limit']:.0%}",
                "max_daily_trades": RISK_DEFAULTS["max_daily_trades"],
                "cooldown_after_loss": f"{RISK_DEFAULTS['cooldown_after_loss']}s",
            })

        st.divider()

        st.subheader("Strategy Configuration")
        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            st.markdown("**Mean Reversion**")
            st.json({
                "min_risk_reward_ratio": f"{RISK_DEFAULTS['min_risk_reward_ratio']}:1",
                "trading_window": "10:00 AM - 2:00 PM ET",
                "regime_detection": "ADX + BB/KC squeeze (5m)",
                "dynamic_targets": "BB middle / VWAP",
                "scale_out": "50% at primary target",
            })

        with strat_col2:
            st.markdown("**Execution**")
            st.json({
                "paper_mode": settings.trading.paper_mode,
                "broker": "Tradovate (Apex)",
                "commission_rt": f"${MES_SPEC['commission_per_side'] * 2:.2f}",
                "slippage": "1.5 ticks/side avg",
                "fill_model": "Next-bar open (no look-ahead)",
                "tick_value": f"${MES_SPEC['tick_value']}/tick",
            })

        st.divider()

        st.subheader("News Blackout Buffers")
        news_col1, news_col2, news_col3 = st.columns(3)
        with news_col1:
            st.markdown("**FOMC**")
            st.caption("30 min before / 60 min after")
        with news_col2:
            st.markdown("**NFP**")
            st.caption("15 min before / 45 min after")
        with news_col3:
            st.markdown("**CPI**")
            st.caption("15 min before / 30 min after")


if __name__ == "__main__":
    main()
