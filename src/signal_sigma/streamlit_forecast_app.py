import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei im data-Verzeichnis
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
CSV_FILENAME = 'NVDA_forecast_eval_metrics_2014-01-01_2025-05-09.csv'
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    return df

# Daten laden
df = load_data(CSV_PATH)

# Aktiensymbol aus Dateinamen extrahieren
target_stock = CSV_FILENAME.split('_')[0]

# Sidebar-Navigation
st.sidebar.title('Navigation')
pages = [
    'Forecast vs Actual',
    'Forecast vs Actual (2025+)',
    'Forecast vs Actual (2018+)',
    'Average Forecast Metrics',
    'Metrics Subplots',
    'Evaluation Metrics',
    'Metrics Plot',
    'Full Signal View'
]
page = st.sidebar.radio('Select Page', pages)

# Helper: Rotate xticklabels
def rotate_xticks(ax):
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Seite: Forecast vs Actual - vollständiger Verlauf
if page == 'Forecast vs Actual':
    st.header(f'{target_stock} Forecast vs Actual - Full History')
    fig, ax = plt.subplots()
    ax.plot(df.index, df['true'], label='True', linewidth=2)
    ax.plot(df.index, df['p50'], label='Median Forecast (p50)', linewidth=2)
    ax.fill_between(df.index, df['p10'], df['p90'], alpha=0.3, label='Confidence Interval (p10–p90)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Target Value')
    ax.grid(True)
    rotate_xticks(ax)
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Seite: Forecast vs Actual ab 2025
elif page == 'Forecast vs Actual (2025+)':
    st.header(f'{target_stock} Forecast vs Actual (from 2025-01-01)')
    df_2025 = df[df.index >= '2025-01-01']
    fig, ax = plt.subplots()
    ax.plot(df_2025.index, df_2025['true'], label='True', linewidth=2)
    ax.plot(df_2025.index, df_2025['p50'], label='Median Forecast (p50)', linewidth=2)
    ax.fill_between(df_2025.index, df_2025['p10'], df_2025['p90'], alpha=0.3, label='Confidence Interval (p10–p90)')
    ax.plot(df_2025.index, df_2025['point_forecast'], linestyle=':', label='Forecast Point', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Target Value')
    ax.grid(True)
    rotate_xticks(ax)
    ax.legend(loc='upper right', ncol=3)
    st.pyplot(fig)

# Seite: Forecast vs Actual ab 2018
elif page == 'Forecast vs Actual (2018+)':
    st.header(f'{target_stock} Forecast vs Actual (from 2018-01-01)')
    df_2018 = df[df.index >= '2018-01-01']
    fig, ax = plt.subplots()
    ax.plot(df_2018.index, df_2018['true'], label='True', linewidth=2)
    ax.plot(df_2018.index, df_2018['p50'], label='Median Forecast (p50)', linewidth=2)
    ax.fill_between(df_2018.index, df_2018['p10'], df_2018['p90'], alpha=0.3, label='Confidence Interval (p10–p90)')
    ax.plot(df_2018.index, df_2018['point_forecast'], linestyle=':', label='Forecast Point', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Target Value')
    ax.grid(True)
    rotate_xticks(ax)
    ax.legend(loc='upper right', ncol=3)
    st.pyplot(fig)

# Seite: Durchschnittliche Forecast-Metriken
elif page == 'Average Forecast Metrics':
    st.header('Average Forecast Metrics')
    avg_metrics = {
        'R2 (Point)': df['r2_point'].mean(),
        'R2 (p50)': df['r2_p50'].mean(),
        'MAE (Point)': df['mae_point'].mean(),
        'MAE (p50)': df['mae_p50'].mean(),
        'Pinball (p10)': df['pinball_p10'].mean(),
        'Pinball (p50)': df['pinball_p50'].mean(),
        'Pinball (p90)': df['pinball_p90'].mean(),
        'Interval Covered (80%)': df['interval_covered_80'].mean()
    }
    fig, ax = plt.subplots()
    bars = ax.bar(list(avg_metrics.keys()), list(avg_metrics.values()))
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center')
    ax.set_ylabel('Metric Value')
    ax.set_title('Average Forecast Metrics')
    rotate_xticks(ax)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Seite: Metrics Subplots
elif page == 'Metrics Subplots':
    st.header('Metrics Overview')
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(df.index, df['mae_p50'], label='MAE (p50)', linewidth=2)
    axs[0].set_title('MAE (p50)')
    axs[0].grid(True)
    axs[0].legend(loc='best')

    axs[1].plot(df.index, df['pinball_p10'], label='Pinball (p10)', linewidth=2)
    axs[1].plot(df.index, df['pinball_p50'], label='Pinball (p50)', linewidth=2)
    axs[1].plot(df.index, df['pinball_p90'], label='Pinball (p90)', linewidth=2)
    axs[1].set_title('Pinball Losses')
    axs[1].grid(True)
    axs[1].legend(loc='best')

    axs[2].plot(df.index, df['interval_covered_80'], label='Interval Covered (80%)', linestyle='--', linewidth=2)
    axs[2].set_title('80% Interval Coverage')
    axs[2].set_xlabel('Date')
    axs[2].grid(True)
    axs[2].legend(loc='best')

    for ax in axs:
        rotate_xticks(ax)
    st.pyplot(fig)

# Seite: Evaluation Metrics über die Zeit
elif page == 'Evaluation Metrics':
    st.header('NVDA Forecast Evaluation Metrics Over Time')
    metric_columns = [
        'mae_point', 'mape_point', 'smape_point', 'rmse_point', 'r2_point',
        'mae_p50', 'mape_p50', 'smape_p50', 'rmse_p50', 'r2_p50'
    ]
    for metric in metric_columns:
        fig, ax = plt.subplots()
        ax.plot(df.index, df[metric], label=metric)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Date')
        ax.set_ylabel(metric)
        ax.grid(True)
        rotate_xticks(ax)
        ax.legend(loc='best')
        st.pyplot(fig)
    st.header('Metrics Summary')
    st.dataframe(df[metric_columns].describe())

# Seite: Neuer Metrics Plot mit Farben und Titeln
elif page == 'Metrics Plot':
    st.header(f'{target_stock} Forecast Evaluation Metrics Plot')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

    dates = df.index
    mae_p50 = df['mae_p50']
    pinball_p10 = df['pinball_p10']
    pinball_p50 = df['pinball_p50']
    pinball_p90 = df['pinball_p90']
    interval_covered_80 = df['interval_covered_80']

    axs[0].plot(dates, mae_p50, label="MAE (p50)", color='green', linewidth=2)
    axs[0].set_ylabel("MAE")
    axs[0].set_title(f"Mean Absolute Error for Median Forecast (p50) -- ({target_stock})")
    axs[0].grid(True)
    axs[0].legend(loc='best')

    axs[1].plot(dates, pinball_p10, label="Pinball Loss (p10)", color='skyblue', linewidth=2)
    axs[1].plot(dates, pinball_p50, label="Pinball Loss (p50)", color='purple', linewidth=2)
    axs[1].plot(dates, pinball_p90,	label="Pinball Loss (p90)", color='brown', linewidth=2)
    axs[1].set_ylabel("Pinball Loss")
    axs[1].set_title(f"Pinball Loss for Quantile Forecasts -- ({target_stock})")
    axs[1].grid(True)
    axs[1].legend(loc='best')

    axs[2].plot(dates, interval_covered_80, label="Interval Covered (80%)", color='black', linestyle='--', linewidth=2)
    axs[2].set_ylabel("Coverage")
    axs[2].set_title(f"80% Prediction Interval Coverage -- ({target_stock})")
    axs[2].set_xlabel("Date")
    axs[2].grid(True)
    axs[2].legend(loc='best')

    for ax in axs:
        rotate_xticks(ax)
    st.pyplot(fig)

# Seite: Vollständige Signalanzeige mit Splits
elif page == 'Full Signal View':
    st.header(f"{target_stock} Forecast vs. True - Full Signal View ({target_stock})")
    full_target_index = df.index
    full_target_vals = df['true']
    forecast_index = df.index
    y_true = df['true']
    y_p50 = df['p50']
    y_p10 = df['p10']
    y_p90 = df['p90']
    y_point = df['point_forecast']

    n = len(df)
    train_end = full_target_index[int(n * 0.7)]
    val_end = full_target_index[int(n * 0.85)]
    test_start = full_target_index[int(n * 0.85)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(full_target_index, full_target_vals, label="Historical Target", color="black", linewidth=2)
    ax.plot(forecast_index,	y_true, label="True (Test)", color="blue", linewidth=2)
    ax.plot(forecast_index, y_p50, label="Forecast Median (p50)", color="magenta", linewidth=2)
    ax.plot(forecast_index, y_p10, label="Forecast Lower (p10)", linestyle="--", color="skyblue")
    ax.plot(forecast_index, y_p90, label="Forecast Upper (p90)", linestyle="--", color="green")
    ax.plot(forecast_index, y_point, label="Forecast Point", linestyle=":", color="orange")
    ax.fill_between(forecast_index, y_p10, y_p90, color="lightgray", alpha=0.4, label="Confidence Interval (p10–p90)")
    ax.axvline(train_end, color="gray", linestyle="--", linewidth=1.5, label="Train End")
    ax.axvline(val_end, color="purple", linestyle="--", linewidth=1.5, label="Validation End")
    ax.axvline(test_start, color="red", linestyle="--", linewidth=2, label="Forecast Start")
    ax.set_title(f"TFT Forecast vs. True - Full Signal View ({target_stock})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Target Value")
    ax.grid(True)
    rotate_xticks(ax)
    ax.legend(loc='best')
    st.pyplot(fig)
