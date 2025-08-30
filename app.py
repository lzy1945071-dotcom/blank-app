# app.py — Quant Terminal (Two-Page: Charts & Strategy) — non-breaking structure
# -------------------------------------------------------
# Key rules satisfied:
# 1) Page 1: ONLY K-line (candles) + optional sub-indicators (toggle incl. Fibonacci default OFF)
# 2) Page 2: Strategies, backtest, scores, reasons, risk mgmt, export
# 3) Keep structure simple & extensible, no destructive refactor
#
# Notes:
# - Data source: yfinance (or CSV upload). If yfinance not available, CSV upload still works.
# - Indicators implemented with numpy/pandas (no TA-Lib dependency).
# - Report export to Excel; PDF optional if reportlab present.
# - This file is self-contained; requirements are in requirements.txt
#
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

# Optional: yfinance for quick demo fetching (user can also upload CSV)
try:
    import yfinance as yf
except Exception:
    yf = None

import plotly.graph_objects as go


# ---------------------------
# Utility & Caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_yf(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    if yf is None or not symbol:
        return pd.DataFrame()
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    df.dropna(inplace=True)
    return df

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"Open","High","Low","Close"}
    if not cols.issubset(set(df.columns)):
        # try lower-case
        alt = {c.title(): c for c in df.columns}
        # attempt rename
        for need in cols:
            if need not in df.columns and need.lower() in df.columns:
                df.rename(columns={need.lower(): need}, inplace=True)
        # recheck
    if not cols.issubset(set(df.columns)):
        raise ValueError("Uploaded data must contain columns: Open, High, Low, Close (Volume optional).")
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df

# ---------------------------
# Indicators (no TA-Lib)
# ---------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-prev_close).abs(), (df["Low"]-prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)
    tr = true_range(df)
    atr_ = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr_.rolling(period).sum() + 1e-12))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr_.rolling(period).sum() + 1e-12))
    dx = 100 * ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) )
    adx_ = dx.rolling(period).mean()
    return adx_

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    # Simplified SuperTrend
    atr_ = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    upperband = hl2 + multiplier * atr_
    lowerband = hl2 - multiplier * atr_

    trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)  # 1 bull, -1 bear
    trend.iloc[0] = hl2.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        prev = trend.iloc[i-1]
        if df["Close"].iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        if direction.iloc[i] == 1:
            trend.iloc[i] = min(lowerband.iloc[i], prev)
        else:
            trend.iloc[i] = max(upperband.iloc[i], prev)
    return trend

def swing_high_low(df: pd.DataFrame, window: int = 20):
    # basic recent swing high/low for Fibonacci
    swing_high = df["High"].rolling(window).max().iloc[-1]
    swing_low = df["Low"].rolling(window).min().iloc[-1]
    return swing_high, swing_low

def fib_levels(high: float, low: float):
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236*diff,
        "38.2%": high - 0.382*diff,
        "50.0%": high - 0.5*diff,
        "61.8%": high - 0.618*diff,
        "78.6%": high - 0.786*diff,
        "100%": low
    }
    return levels

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_candles_with_indicators(df: pd.DataFrame, show_volume=True, show_rsi=False, show_macd=False,
                                 show_bbands=False, show_adx=False, show_supertrend=False,
                                 show_fib=False, fib_window=100):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    # Bollinger on price
    if show_bbands:
        up, ma, low = bollinger_bands(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=up, mode="lines", name="BB Upper"))
        fig.add_trace(go.Scatter(x=df.index, y=ma, mode="lines", name="BB Mid"))
        fig.add_trace(go.Scatter(x=df.index, y=low, mode="lines", name="BB Lower"))

    # Supertrend on price
    if show_supertrend:
        st_line = supertrend(df)
        fig.add_trace(go.Scatter(x=df.index, y=st_line, mode="lines", name="SuperTrend"))

    # Fibonacci (default OFF)
    if show_fib and len(df) >= fib_window:
        h, l = swing_high_low(df.iloc[-fib_window:], window=min(20, fib_window))
        levels = fib_levels(h, l)
        for label, lvl in levels.items():
            fig.add_trace(go.Scatter(x=df.index, y=[lvl]*len(df), mode="lines",
                                     name=f"Fib {label}", line=dict(dash="dot")))

    # Layout
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=20, r=20, t=40, b=20))

    # Subcharts (volume / indicators) as separate figures in Streamlit
    figs = [fig]

    if show_volume and "Volume" in df.columns:
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
        f2.update_layout(height=120, margin=dict(l=20, r=20, t=10, b=20))
        figs.append(f2)

    if show_rsi:
        r = rsi(df["Close"])
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=df.index, y=r, name="RSI"))
        f3.add_hline(y=70, line_dash="dot")
        f3.add_hline(y=30, line_dash="dot")
        f3.update_layout(height=160, margin=dict(l=20, r=20, t=10, b=20))
        figs.append(f3)

    if show_macd:
        m_line, s_line, hist = macd(df["Close"])
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=df.index, y=m_line, name="MACD"))
        f4.add_trace(go.Scatter(x=df.index, y=s_line, name="Signal"))
        f4.add_trace(go.Bar(x=df.index, y=hist, name="Hist"))
        f4.update_layout(height=160, margin=dict(l=20, r=20, t=10, b=20))
        figs.append(f4)

    if show_adx:
        a = adx(df)
        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=df.index, y=a, name="ADX"))
        f5.add_hline(y=25, line_dash="dot")
        f5.update_layout(height=160, margin=dict(l=20, r=20, t=10, b=20))
        figs.append(f5)

    return figs


# ---------------------------
# Strategy, Scoring, Backtest
# ---------------------------
def indicator_signals(df: pd.DataFrame):
    # Compute various indicator states for scoring
    signals = {}
    close = df["Close"]
    # RSI zones
    r = rsi(close).iloc[-1]
    signals["RSI"] = r

    # MACD cross
    m_line, s_line, hist = macd(close)
    signals["MACD_hist"] = hist.iloc[-1]
    signals["MACD_above"] = (m_line.iloc[-1] > s_line.iloc[-1])

    # Bollinger position
    up, ma, low = bollinger_bands(close)
    last = close.iloc[-1]
    if last > up.iloc[-1]:
        bb_pos = "above_upper"
    elif last < low.iloc[-1]:
        bb_pos = "below_lower"
    elif last >= ma.iloc[-1]:
        bb_pos = "above_mid"
    else:
        bb_pos = "below_mid"
    signals["BB_pos"] = bb_pos

    # ADX trend strength
    a = adx(df)
    signals["ADX"] = a.iloc[-1]

    # Supertrend direction
    st_line = supertrend(df)
    signals["SuperTrend_bull"] = (close.iloc[-1] >= st_line.iloc[-1])

    # ATR for risk sizing
    signals["ATR"] = atr(df).iloc[-1]
    return signals

def score_long_short(signals: dict):
    # Simple heuristic scoring: 0-100
    long_score = 50
    short_score = 50

    # RSI contribution
    r = signals.get("RSI", 50)
    if r < 30: long_score += 15; short_score -= 10
    if r > 70: long_score -= 10; short_score += 15

    # MACD
    if signals.get("MACD_above", False): long_score += 10; short_score -= 10
    else: short_score += 10; long_score -= 10

    # BB position
    bb = signals.get("BB_pos", "above_mid")
    if bb == "below_lower":
        long_score += 10; short_score -= 5
    elif bb == "above_upper":
        short_score += 10; long_score -= 5

    # ADX strength
    adx_v = signals.get("ADX", 15)
    if adx_v >= 25:
        if signals.get("MACD_above", False) and signals.get("SuperTrend_bull", False):
            long_score += 10
        if (not signals.get("MACD_above", False)) and (not signals.get("SuperTrend_bull", True)):
            short_score += 10

    # SuperTrend
    if signals.get("SuperTrend_bull", False): long_score += 5
    else: short_score += 5

    long_score = int(np.clip(long_score, 0, 100))
    short_score = int(np.clip(short_score, 0, 100))
    return long_score, short_score

def generate_reasoning(signals: dict) -> str:
    parts = []
    r = signals["RSI"]
    parts.append(f"RSI={r:.1f} ({'oversold' if r<30 else 'overbought' if r>70 else 'neutral'})")
    parts.append(f"MACD {'bullish' if signals['MACD_above'] else 'bearish'} (hist {signals['MACD_hist']:.3f})")
    parts.append(f"Bollinger position={signals['BB_pos']}")
    parts.append(f"ADX={signals['ADX']:.1f} ({'trending' if signals['ADX']>=25 else 'weak'})")
    parts.append(f"SuperTrend={'bullish' if signals['SuperTrend_bull'] else 'bearish'}")
    return "; ".join(parts)

def backtest_simple(df: pd.DataFrame, params: dict):
    # Simple rule: MACD cross + RSI filter + SuperTrend direction
    m_line, s_line, _ = macd(df["Close"], fast=params.get("macd_fast",12), slow=params.get("macd_slow",26), signal=params.get("macd_signal",9))
    r = rsi(df["Close"], period=params.get("rsi_period",14))
    st_line = supertrend(df, period=params.get("st_period",10), multiplier=params.get("st_mult",3.0))

    long_entry = (m_line > s_line) & (r > params.get("rsi_long_th", 45)) & (df["Close"] > st_line)
    long_exit  = (m_line < s_line) | (r < params.get("rsi_exit_th", 40)) | (df["Close"] < st_line)

    pos = 0
    entries, exits, pnl = [], [], []
    entry_px = 0.0

    for i in range(1, len(df)):
        if pos == 0 and long_entry.iloc[i-1] and not long_entry.iloc[i-2] if i>=2 else long_entry.iloc[i-1]:
            pos = 1
            entry_px = df["Close"].iloc[i]
            entries.append((df.index[i], entry_px))
        elif pos == 1 and (long_exit.iloc[i]):
            pos = 0
            exit_px = df["Close"].iloc[i]
            exits.append((df.index[i], exit_px))
            pnl.append((exit_px - entry_px) / entry_px)

    # close any open trade at last
    if pos == 1:
        exit_px = df["Close"].iloc[-1]
        exits.append((df.index[-1], exit_px))
        pnl.append((exit_px - entry_px) / entry_px)

    trades = pd.DataFrame({
        "EntryTime": [e[0] for e in entries],
        "EntryPrice": [e[1] for e in entries],
        "ExitTime": [x[0] for x in exits],
        "ExitPrice": [x[1] for x in exits],
    })
    if not trades.empty:
        trades["Return"] = (trades["ExitPrice"] - trades["EntryPrice"]) / trades["EntryPrice"]

    # Equity curve (1 unit each)
    equity = (1 + pd.Series([0] + [r for r in trades["Return"]] if not trades.empty else [0])).cumprod()

    # Metrics
    if not trades.empty:
        win_rate = (trades["Return"] > 0).mean()
        avg_ret = trades["Return"].mean()
        profit_factor = trades.loc[trades["Return"]>0, "Return"].sum() / abs(trades.loc[trades["Return"]<=0, "Return"].sum() + 1e-12)
        max_dd = (equity / equity.cummax() - 1).min()
    else:
        win_rate, avg_ret, profit_factor, max_dd = 0.0, 0.0, 0.0, 0.0

    metrics = {
        "Trades": int(0 if trades.empty else len(trades)),
        "WinRate": float(win_rate),
        "AvgReturnPerTrade": float(avg_ret),
        "ProfitFactor": float(profit_factor),
        "MaxDrawdown": float(max_dd),
        "FinalEquity": float(equity.iloc[-1])
    }
    return trades, equity, metrics

def position_size(balance: float, atr_value: float, risk_pct: float = 0.01, atr_mult_sl: float = 2.0) -> float:
    # Risk position sizing: risk fixed % of balance with ATR-based stop
    risk_amount = balance * risk_pct
    stop_distance = atr_value * atr_mult_sl
    if stop_distance <= 0:
        return 0.0
    size = risk_amount / stop_distance
    return max(size, 0.0)


# ---------------------------
# Layout
# ---------------------------
st.set_page_config(page_title="Quant Terminal — Charts & Strategy", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", ["Charts", "Strategy & Backtest"], index=0)

# Data input
st.sidebar.header("Data")
data_source = st.sidebar.selectbox("Source", ["Yahoo Finance", "Upload CSV"], index=0)

if data_source == "Yahoo Finance":
    symbol = st.sidebar.text_input("Symbol", value="AAPL")
    period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y","10y","max"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","60m","1d","1wk","1mo"], index=5)
    df = load_yf(symbol, period, interval)
else:
    up = st.sidebar.file_uploader("Upload CSV (columns: Date,Open,High,Low,Close,Volume)", type=["csv"])
    if up:
        df = pd.read_csv(up, parse_dates=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
    else:
        df = pd.DataFrame()

if not df.empty:
    df = ensure_ohlcv(df)

# ---------------------------
# PAGE 1: Charts (K-line + subcharts only)
# ---------------------------
if page == "Charts":
    st.title("📈 Charts")
    if df.empty:
        st.info("Load data on the left to display charts.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_volume = st.checkbox("Show Volume", value=True)
        show_bbands = st.checkbox("Bollinger Bands", value=False)
    with col2:
        show_rsi = st.checkbox("RSI", value=False)
        show_macd = st.checkbox("MACD", value=False)
    with col3:
        show_adx = st.checkbox("ADX", value=False)
        show_supertrend = st.checkbox("SuperTrend", value=False)
    with col4:
        show_fib = st.checkbox("Fibonacci (default OFF)", value=False)
        fib_window = st.number_input("Fib window (bars)", min_value=50, max_value=2000, value=200, step=10)

    figs = plot_candles_with_indicators(df,
                                        show_volume=show_volume,
                                        show_rsi=show_rsi,
                                        show_macd=show_macd,
                                        show_bbands=show_bbands,
                                        show_adx=show_adx,
                                        show_supertrend=show_supertrend,
                                        show_fib=show_fib,
                                        fib_window=fib_window)

    for f in figs:
        st.plotly_chart(f, use_container_width=True)


# ---------------------------
# PAGE 2: Strategy & Backtest
# ---------------------------
if page == "Strategy & Backtest":
    st.title("🧠 Strategy, Scoring & Backtest")
    if df.empty:
        st.info("Load data on the left to run strategies.")
        st.stop()

    st.subheader("Signals & Scores")
    sig = indicator_signals(df)
    long_s, short_s = score_long_short(sig)
    reasoning = generate_reasoning(sig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Long Score", long_s)
    c2.metric("Short Score", short_s)
    c3.metric("ATR (volatility)", f"{sig['ATR']:.3f}")

    st.write("**Signal reasoning**:", reasoning)

    st.subheader("Risk & Sizing")
    colA, colB, colC = st.columns(3)
    with colA:
        balance = st.number_input("Account Balance", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
    with colB:
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0
    with colC:
        atr_mult = st.slider("Stop (ATR multiples)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    size = position_size(balance, sig["ATR"], risk_pct=risk_pct, atr_mult_sl=atr_mult)
    st.write(f"**Suggested Position Size**: {size:.4f} units (risk {risk_pct*100:.1f}% of equity with {atr_mult}×ATR stop)")

    st.subheader("Backtest")
    with st.expander("Parameters", expanded=True):
        p1, p2, p3, p4, p5 = st.columns(5)
        macd_fast = p1.number_input("MACD Fast", 2, 50, 12, 1)
        macd_slow = p2.number_input("MACD Slow", 5, 100, 26, 1)
        macd_signal = p3.number_input("MACD Signal", 2, 50, 9, 1)
        rsi_period = p4.number_input("RSI Period", 2, 50, 14, 1)
        st_period = p5.number_input("SuperTrend Period", 5, 50, 10, 1)

        q1, q2 = st.columns(2)
        rsi_long_th = q1.slider("RSI Long Threshold", 10, 70, 45, 1)
        rsi_exit_th = q2.slider("RSI Exit Threshold", 10, 70, 40, 1)

        st_mult = st.slider("SuperTrend Multiplier", 1.0, 5.0, 3.0, 0.1)

    params = dict(macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                  rsi_period=rsi_period, rsi_long_th=rsi_long_th, rsi_exit_th=rsi_exit_th,
                  st_period=st_period, st_mult=st_mult)

    trades, equity, metrics = backtest_simple(df, params)

    st.write("**Metrics**")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Trades", metrics["Trades"])
    m2.metric("Win Rate", f"{metrics['WinRate']*100:.1f}%")
    m3.metric("Avg Return/Trade", f"{metrics['AvgReturnPerTrade']*100:.2f}%")
    m4.metric("Profit Factor", f"{metrics['ProfitFactor']:.2f}")
    m5.metric("Max Drawdown", f"{metrics['MaxDrawdown']*100:.2f}%")
    m6.metric("Final Equity (1u)", f"{metrics['FinalEquity']:.3f}")

    # Equity curve
    if not trades.empty:
        eq = go.Figure()
        eq.add_trace(go.Scatter(x=range(len(equity)), y=equity, name="Equity"))
        eq.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(eq, use_container_width=True)

        st.dataframe(trades, use_container_width=True)

    # Export report (Excel)
    st.subheader("Export Report")
    report_name = f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if st.button("Generate Excel Report"):
        buf = pd.ExcelWriter(report_name, engine="xlsxwriter")
        # Save inputs
        pd.DataFrame([params]).to_excel(buf, index=False, sheet_name="Params")
        # Save metrics
        pd.DataFrame([metrics]).to_excel(buf, index=False, sheet_name="Metrics")
        # Save last signals
        pd.DataFrame([indicator_signals(df)]).to_excel(buf, index=False, sheet_name="Signals")
        # Save trades
        if not trades.empty:
            trades.to_excel(buf, index=False, sheet_name="Trades")
        # Save last 300 bars
        df.tail(300).to_excel(buf, sheet_name="Last300")
        buf.close()
        with open(report_name, "rb") as f:
            st.download_button("Download Excel Report", f, file_name=report_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption("Tip: Use the Charts page for pure visualization; all strategy logic remains here.")
