"""Streamlit dashboard for the trading assistant.

Displays live candlestick charts, indicators, positions, P&L,
trade journal, and system health status.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text

from src.config import settings
from src.core.database import get_duckdb_connection, get_session, get_sqlite_engine


# ── Page Config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data Loading ────────────────────────────────────────────────────

@st.cache_resource
def _get_duckdb():
    return get_duckdb_connection()


@st.cache_resource
def _get_sqlite():
    return get_sqlite_engine()


def load_bars(conn: duckdb.DuckDBPyConnection, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
    """Load recent bars from DuckDB."""
    table = f"bars_{timeframe}"
    try:
        df = conn.execute(
            f"SELECT * FROM {table} WHERE symbol = 'MES' ORDER BY timestamp DESC LIMIT ?",
            [limit],
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
               WHERE symbol = 'MES' AND timeframe = ?
               ORDER BY timestamp DESC LIMIT ?""",
            [timeframe, limit],
        ).fetchdf()
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_trades(engine, limit: int = 50) -> pd.DataFrame:
    """Load recent trades from SQLite."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM trades ORDER BY created_at DESC LIMIT :limit"),
                conn,
                params={"limit": limit},
            )
        return df
    except Exception:
        return pd.DataFrame()


def load_daily_summary(engine) -> pd.DataFrame:
    """Load daily summaries."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM daily_summary ORDER BY date DESC LIMIT 30"),
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()


def load_equity_snapshots(engine, limit: int = 500) -> pd.DataFrame:
    """Load equity snapshots."""
    try:
        with engine.connect() as conn:
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


# ── Chart Building ──────────────────────────────────────────────────

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
                # Overbought/oversold lines
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


# ── Main App ────────────────────────────────────────────────────────

def main():
    duck = _get_duckdb()
    sqlite = _get_sqlite()

    # Sidebar
    with st.sidebar:
        st.title("Trading Assistant")
        st.divider()

        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mode", "PAPER" if settings.trading.paper_mode else "LIVE")
        with col2:
            st.metric("Symbol", settings.trading.symbol)

        st.metric("Account Equity", f"${settings.trading.account_equity:,.2f}")

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
    tab_chart, tab_journal, tab_analytics, tab_health = st.tabs(
        ["Chart", "Trade Journal", "Analytics", "System Health"]
    )

    with tab_chart:
        bars_df = load_bars(duck, timeframe, bar_count)
        indicators_df = load_indicators(duck, timeframe, bar_count)

        if bars_df.empty:
            st.info("No market data yet. Connect to IB and start receiving bars.")
        else:
            fig = build_price_chart(bars_df, indicators_df)
            st.plotly_chart(fig, use_container_width=True)

            # Current price info
            if not bars_df.empty:
                last = bars_df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Price", f"{last['close']:.2f}")
                col2.metric("High", f"{last['high']:.2f}")
                col3.metric("Low", f"{last['low']:.2f}")
                col4.metric("Volume", f"{last['volume']:,}")

    with tab_journal:
        trades_df = load_trades(sqlite)
        if trades_df.empty:
            st.info("No trades recorded yet.")
        else:
            # Summary metrics
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

            # Trade table
            display_cols = [
                c for c in ["entry_time", "direction", "entry_price", "exit_price",
                            "pnl_dollars", "strategy", "status", "signal_confidence"]
                if c in trades_df.columns
            ]
            st.dataframe(trades_df[display_cols] if display_cols else trades_df, use_container_width=True)

    with tab_analytics:
        daily_df = load_daily_summary(sqlite)
        equity_df = load_equity_snapshots(sqlite)

        if equity_df.empty and daily_df.empty:
            st.info("No analytics data yet. Start paper trading to see performance metrics.")
        else:
            if not equity_df.empty:
                fig = go.Figure()
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

    with tab_health:
        st.subheader("Service Health")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IB Connection", "Check pending...")
        col2.metric("DuckDB", "Check pending...")
        col3.metric("SQLite", "Check pending...")
        col4.metric("Ollama", "Check pending...")

        st.subheader("Risk Status")
        st.json({
            "paper_mode": settings.trading.paper_mode,
            "risk_config": {
                "max_risk_per_trade": "1.5%",
                "daily_loss_limit": "3%",
                "weekly_loss_limit": "6%",
                "max_daily_trades": 10,
                "min_risk_reward": "1:2",
                "max_concurrent_positions": 2,
                "always_use_stop_loss": True,
            },
        })


if __name__ == "__main__":
    main()
