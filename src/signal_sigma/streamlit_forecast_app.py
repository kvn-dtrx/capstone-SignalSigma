import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from pandas import Timestamp
import signal_sigma.config.cfg as cfg

# === Page config ===
st.set_page_config(layout="wide", page_title="Stock Forecast Visualizer")

# === Stock Options ===
stock_options = {
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Apple (AAPL)": "AAPL",
    "Meta (META)": "META"
}

# === Sidebar Configuration ===
st.sidebar.header("⚙️ Configuration")

selected_display = st.sidebar.selectbox("Choose a stock:", list(stock_options.keys()), index=1)
stock = stock_options[selected_display]

path_stock = "/Users/payamoghtanem/capstone-SignalSigma/data/Stock_market_data"
raw_path = f"{path_stock}/{stock}_reduced_dataset_2014-01-01_2025-05-14.csv"

# Load full raw_df temporarily to get available dates for this stock
if not os.path.exists(raw_path):
    st.error(f"Data file for {stock} not found.")
    st.stop()

raw_df_full = pd.read_csv(raw_path, parse_dates=["date"], index_col="date").sort_index()
min_date = raw_df_full.index.min()
max_date = raw_df_full.index.max()

# Date range selection dynamically based on raw_df date coverage
# Fixed start_date
start_date = pd.to_datetime("2014-01-01")

# Dynamic end_date selection
st.sidebar.markdown("### Select End Date")
end_date = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=start_date,
    max_value=max_date
)
end_date = pd.to_datetime(end_date)

# Plot type
plot_mode = st.sidebar.radio(
    "Select Plot Type",
    ["Forecast", "Daily Metric", "Summary Metrics", "All Stocks Actual"]
)

# === Load Data for Selected Stock & Date Range ===
forecast_path = f"{path_stock}/{stock}_forecast_eval_metrics_{start_date.date()}_{end_date.date()}.csv"
raw_path = f"{path_stock}/{stock}_reduced_dataset_{start_date.date()}_{end_date.date()}.csv"

if not (os.path.exists(forecast_path) and os.path.exists(raw_path)):
    st.error(f"Data files for {stock} not found for selected time range.")
    st.stop()

raw_df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date").sort_index()
forecast_df = pd.read_csv(forecast_path, parse_dates=["date"], index_col="date").sort_index()

forecast_df_index = forecast_df.index
y_actual = raw_df["target"]
y_p10 = forecast_df["p10"]
y_p50 = forecast_df["p50"]
y_p90 = forecast_df["p90"]
y_point = forecast_df["point_forecast"]
test_start = forecast_df_index[0]

# === Forecast Plot ===
if plot_mode == "Forecast":
    st.title("📈 Stock Forecast Visualization")
    st.markdown("""
    This dashboard visualizes forecasted stock prices using two prediction approaches:

    - **Quantile Forecasting (p10, p50, p90)**: These are estimates of a range of possible outcomes. 
      - p10 = conservative/lower estimate
      - p50 = typical/median estimate
      - p90 = optimistic/upper estimate
      - The shaded area between p10 and p90 shows the uncertainty interval. Narrower is more confident.

    - **Single Point Forecast**: A single best-guess prediction.

    ### How to interpret:
    - If actual price stays within the p10–p90 range → the model captures uncertainty well.
    - If actual price tracks close to p50 → the median forecast is well-calibrated.
    - If point forecast tracks better than p50 → point modeling is better suited.
    """)

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=raw_df.index, y=y_actual, name="Actual Price", line=dict(color="orange", width=2)))
    forecast_fig.add_trace(go.Scatter(x=forecast_df_index, y=y_p50, name="p50 Quantile Forecast", line=dict(color="magenta", width=2)))
    forecast_fig.add_trace(go.Scatter(x=forecast_df_index, y=y_point, name="Single Point Forecast", line=dict(color="blue", dash="dot", width=2)))
    forecast_fig.add_trace(go.Scatter(x=forecast_df_index, y=y_p90, name="p90 Quantile Forecast", line=dict(color="green", width=1)))
    forecast_fig.add_trace(go.Scatter(x=forecast_df_index, y=y_p10, name="p10 Quantile Forecast", line=dict(color="red", width=1)))
    forecast_fig.add_trace(go.Scatter(x=forecast_df_index, y=y_p10, name="p10–p90 Confidence Interval", fill='tonexty',
                                      fillcolor='rgba(173,216,230,0.3)', line=dict(width=0), hoverinfo="skip"))
    forecast_fig.add_shape(type="line", x0=test_start, x1=test_start, y0=0, y1=1, xref='x', yref='paper',
                           line=dict(color="red", dash="dash", width=1))
    forecast_fig.add_annotation(x=test_start, y=1.02, xref="x", yref="paper", showarrow=False,
                                text="Forecast Start", font=dict(color="red", size=11))
    forecast_fig.update_layout(title=f"{stock} Forecast vs Actual", xaxis_title="Date", yaxis_title="Price (US$)",
                               hovermode="x unified", template="plotly_white", height=750,
                               legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    st.plotly_chart(forecast_fig, use_container_width=True)

# === All Stocks Actual Plot ===
elif plot_mode == "All Stocks Actual":
    st.title("📊 All Stocks: Actual Price Trends")
    stock_data = {}
    for label, symbol in stock_options.items():
        path = f"{path_stock}/{symbol}_reduced_dataset_2014-01-01_2025-05-14.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[(df["date"] >= str(start_date)) & (df["date"] <= str(end_date))]
            df["Stock"] = label
            df = df[["date", "target", "Stock"]].rename(columns={"target": "Actual Price"})
            stock_data[label] = df

    if stock_data:
        all_data = pd.concat(stock_data.values(), ignore_index=True)
        fig_all = px.line(all_data, x="date", y="Actual Price", color="Stock",
                          title="Actual Stock Prices Across All Companies")
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.warning("No valid stock data found for any of the companies.")
