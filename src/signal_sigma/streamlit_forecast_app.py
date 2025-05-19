import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from pandas import Timestamp
import signal_sigma.config.cfg as cfg

# === Stock Selection ===
stock_options = {
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Apple (AAPL)": "AAPL",
    "Meta (META)": "META"
}
selected_display = st.selectbox("Choose a stock:", list(stock_options.keys()), index=1)
stock = stock_options[selected_display]

# === Configuration ===
start_date = "2014-01-01"
end_date = "2025-05-14"
path_stock = "/Users/payamoghtanem/capstone-SignalSigma/data/Stock_market_data"

forecast_path = f"{path_stock}/{stock}_forecast_eval_metrics_{start_date}_{end_date}.csv"
raw_path = f"{path_stock}/{stock}_reduced_dataset_{start_date}_{end_date}.csv"

if not (os.path.exists(forecast_path) and os.path.exists(raw_path)):
    st.error(f"Data files for {stock} not found. Please check the directory and file naming.")
    st.stop()

# === Load Data ===
raw_df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date").sort_index()
forecast_df = pd.read_csv(forecast_path, parse_dates=["date"], index_col="date").sort_index()

# === Forecast Values ===
forecast_df_index = forecast_df.index
y_p10 = forecast_df["p10"]
y_p50 = forecast_df["p50"]
y_p90 = forecast_df["p90"]
y_point = forecast_df["point_forecast"]
y_actual = raw_df["target"]
test_start = forecast_df_index[0]

# === UI ===
st.title(f"📈 {stock} Forecast vs Target")

col1, col2 = st.columns(2)
show_markers = col1.checkbox("Show Forecast Start Marker", value=True)
compare_forecasts = col2.checkbox("Compare p50 vs Point Forecast", value=False)

# === Forecast Plot ===
fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=raw_df.index, y=y_actual,
    name="Actual Price", mode="lines",
    line=dict(color="orange", width=2)
))

# p50 Forecast
fig.add_trace(go.Scatter(
    x=forecast_df_index, y=y_p50,
    name="Quantile Forecast (p50)", mode="lines",
    line=dict(color="magenta", width=2)
))

# Optional point forecast
if compare_forecasts:
    fig.add_trace(go.Scatter(
        x=forecast_df_index, y=y_point,
        name="Single Point Forecast", mode="lines",
        line=dict(color="blue", dash="dot", width=2)
    ))

# Confidence Bounds (visible for debugging)
fig.add_trace(go.Scatter(
    x=forecast_df_index, y=y_p90,
    name="Quantile Forecast Upper (p90)", mode="lines",
    line=dict(color="green", width=1)
))

fig.add_trace(go.Scatter(
    x=forecast_df_index, y=y_p10,
    name="Quantile Forecast Lower (p10)", mode="lines",
    line=dict(color="red", width=1)
))

fig.add_trace(go.Scatter(
    x=forecast_df_index, y=y_p10,
    name="Confidence Interval (p10–p90)", mode="lines",
    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)',
    line=dict(width=0), hoverinfo="skip"
))

# Marker for forecast start
if show_markers:
    fig.add_shape(
        type="line", x0=test_start, x1=test_start, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color="red", dash="dash", width=1)
    )
    fig.add_annotation(
        x=test_start, y=1.02, xref="x", yref="paper",
        showarrow=False, text="Forecast Start",
        font=dict(color="red", size=11)
    )

fig.update_layout(
    title=f"{stock} Forecast vs Actual Price",
    xaxis_title="Date", yaxis_title="Stock Price (US$)",
    hovermode="x unified", template="plotly_white",
    margin=dict(t=80, b=120),
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    height=750, width=1000
)
st.plotly_chart(fig, use_container_width=False)

# === Daily Metrics Viewer ===
st.subheader("📊 Daily Forecast Metrics (Time Series)")

daily_metrics = [
    'mae_point', 'r2_point',
    'mae_p50', 'r2_p50',
    'pinball_p10', 'pinball_p50', 'pinball_p90'
]
selected_metric = st.selectbox("Select metric to plot:", daily_metrics)

metric_fig = go.Figure()
metric_fig.add_trace(go.Scatter(
    x=forecast_df_index, y=forecast_df[selected_metric],
    mode="lines+markers", name=selected_metric,
    line=dict(color="darkblue", width=2)
))
metric_fig.update_layout(
    title=f"Daily {selected_metric}",
    xaxis_title="Date", yaxis_title=selected_metric,
    template="plotly_white", height=400, width=1000, margin=dict(t=60, b=60)
)
st.plotly_chart(metric_fig, use_container_width=False)

# === Summary Metrics Bar Chart ===
st.subheader("📋 Average Forecast Metrics Summary")

avg_metrics = forecast_df[daily_metrics].mean().sort_values()
summary_fig = go.Figure()
summary_fig.add_trace(go.Bar(
    x=avg_metrics.index, y=avg_metrics.values,
    marker=dict(color="teal")
))
summary_fig.update_layout(
    title="Average Metrics Over Forecast Window",
    xaxis_title="Metric", yaxis_title="Average Value",
    xaxis_tickangle=-45, height=500, width=1000,
    template="plotly_white", margin=dict(t=80, b=120)
)
st.plotly_chart(summary_fig, use_container_width=False)
