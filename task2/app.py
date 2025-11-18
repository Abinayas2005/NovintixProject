# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Household Power - Analytics", layout="wide", page_icon="⚡")

# ---------- Helpers ----------
@st.cache_data
def download_ucidata():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    try:
        raw = pd.read_csv(url, compression='zip', sep=';', nrows=5)  # quick check
    except Exception:
        # fallback: try requests (some environments block direct read)
        import requests, zipfile, io
        r = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = [n for n in z.namelist() if n.endswith('.txt')][0]
        df_bytes = z.read(fname)
        return BytesIO(df_bytes)
    # if the quick read above worked, just return the url string for pandas to use
    return url

def load_data(filelike, nrows=None):
    """
    Loads the UCI electricity dataset from a file-like object or URL.
    Handles separator and missing values ('?').
    """
    # The dataset uses semicolon separator and "?" for missing
    df = pd.read_csv(filelike,
                     sep=';',
                     header=0,
                     low_memory=False,
                     na_values=['?'],
                     nrows=nrows)
    # Combine Date and Time into datetime
    # Original Date format: dd/mm/yyyy
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    # Convert numeric columns to float (they may be strings)
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                    'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # set index
    df = df.set_index('Datetime').sort_index()
    # drop redundant columns
    df = df.drop(columns=['Date','Time'])
    return df

def hourly_resample(df):
    """Resample to hourly mean (useful for next-hour forecasting)."""
    hourly = df.resample('H').mean()
    # Interpolate small gaps
    hourly = hourly.interpolate(limit=4)
    return hourly

def make_window_features(series, window=24):
    """
    series: pd.Series (hourly)
    returns DataFrame X (lag features) and y (next hour)
    """
    data = {}
    for i in range(window):
        data[f'lag_{i+1}'] = series.shift(i+1)
    X = pd.DataFrame(data)
    y = series.shift(0)  # current hour aligned with lags (we will predict current from previous)
    # We'll predict next hour, so shift y up by -1
    y = y.shift(-1)
    # drop rows with NaNs
    valid = X.join(y.rename('target')).dropna()
    X = valid.drop(columns=['target'])
    y = valid['target']
    return X, y

def train_test_time_split(X, y, test_size=0.2):
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true.replace(0, np.nan)))) * 100
    return mae, rmse, mape

# ---------- UI ----------
st.title("⚡ Household Electric Power — Full Analysis App")
st.markdown(
    """
    This app performs:
    - Task 1: EDA (time-series trend + missing/abnormal detection + hourly/daily patterns)
    - Task 2: Time-series forecasting (next-hour Global_active_power)
    - Task 3: Anomaly detection & clustering of daily consumption profiles
    - Task 4: Simple rule-based consumption category generator
    """
)

st.sidebar.header("Data Input")
use_download = st.sidebar.checkbox("Download UCI dataset automatically", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV / TXT file (semicolon separated)", type=["csv","txt","zip"])

# Load dataset
df_load_state = st.sidebar.empty()
if use_download and uploaded_file is None:
    df_load_state.info("Downloading dataset from UCI (may take a little while)...")
    source = download_ucidata()
    try:
        # pandas can read directly from the url if returned
        df_raw = load_data(source)
    except Exception as e:
        st.sidebar.error("Automatic download failed. Please upload the dataset manually.")
        st.stop()
    df_load_state.success("Downloaded and loaded dataset.")
elif uploaded_file is not None:
    df_load_state.info("Loading uploaded file...")
    try:
        if isinstance(uploaded_file, BytesIO):
            df_raw = load_data(uploaded_file)
        else:
            # streamlit gives UploadedFile object with .read()
            bytes_io = BytesIO(uploaded_file.read())
            # If it's a zip, reading BytesIO will work as well
            try:
                df_raw = load_data(bytes_io)
            except Exception:
                # fallback to pass name
                df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
    df_load_state.success("Uploaded file loaded.")
else:
    st.info("Choose to download from UCI or upload the dataset to begin.")
    st.stop()

# show overview
st.subheader("Data snapshot")
st.write("Rows:", df_raw.shape[0], " | Columns:", df_raw.shape[1])
st.dataframe(df_raw.head())

# ---------- Task 1: EDA ----------
st.header("Task 1 — EDA: Time-series and patterns")

if st.button("Run EDA"):
    st.subheader("1. Time-series trend: Global_active_power (original frequency)")
    fig, ax = plt.subplots(1,1, figsize=(14,4))
    # plot a limited timeframe for speed (if too long, sample)
    plot_series = df_raw['Global_active_power'].dropna()
    if len(plot_series) > 50000:
        plot_series = plot_series.iloc[:50000]
        ax.set_title("Global_active_power (first 50k rows shown)")
    sns.lineplot(x=plot_series.index, y=plot_series.values, ax=ax)
    ax.set_ylabel("Global_active_power (kW)")
    st.pyplot(fig)

    st.subheader("2. Missing / Abnormal readings")
    missing_count = df_raw['Global_active_power'].isna().sum()
    st.write(f"Missing Global_active_power values: **{missing_count}**")
    st.write("Abnormal readings (negative or extremely large values):")
    abnormal = df_raw[(df_raw['Global_active_power'] < 0) | (df_raw['Global_active_power'] > df_raw['Global_active_power'].quantile(0.999))]
    st.write(f"Found {len(abnormal)} abnormal rows (showing up to 100):")
    st.dataframe(abnormal.head(100))

    st.subheader("3. Hourly and daily patterns")
    st.write("Resampling original data to hourly mean for pattern analysis...")
    hourly = hourly_resample(df_raw)
    st.write("Hourly series sample:")
    st.line_chart(hourly['Global_active_power'].dropna().iloc[:168])  # first 7 days

    st.write("Mean consumption by hour of day (0-23):")
    hourly['hour'] = hourly.index.hour
    hourly_by_hour = hourly.groupby('hour')['Global_active_power'].mean()
    fig2, ax2 = plt.subplots(1,1,figsize=(9,4))
    sns.barplot(x=hourly_by_hour.index, y=hourly_by_hour.values, ax=ax2)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Average Global_active_power (kW)")
    st.pyplot(fig2)

    st.write("Mean consumption by day of week:")
    hourly['dow'] = hourly.index.dayofweek
    dow = hourly.groupby('dow')['Global_active_power'].mean()
    fig3, ax3 = plt.subplots(1,1,figsize=(9,4))
    sns.barplot(x=dow.index, y=dow.values, ax=ax3)
    ax3.set_xlabel("Day of week (0=Mon)")
    ax3.set_ylabel("Average Global_active_power (kW)")
    st.pyplot(fig3)

# ---------- Task 2: Forecasting ----------
st.header("Task 2 — Time-series Forecasting (next-hour Global_active_power)")

forecast_run = st.button("Run Forecasting")
if forecast_run:
    with st.spinner("Preparing hourly data and windowed features..."):
        hourly = hourly_resample(df_raw)['Global_active_power'].dropna()
        st.write("Hourly data length:", len(hourly))
        # build window features (previous 24 hours -> predict next hour)
        X, y = make_window_features(hourly, window=24)
        st.write("Feature matrix shape:", X.shape)
        # train-test split in time
        X_train, X_test, y_train, y_test = train_test_time_split(X, y, test_size=0.2)

    with st.spinner("Training RandomForestRegressor..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    with st.spinner("Evaluating..."):
        y_pred = model.predict(X_test)
        mae, rmse, mape = compute_metrics(y_test, y_pred)
        st.metric("MAE", f"{mae:.4f} kW")
        st.metric("RMSE", f"{rmse:.4f} kW")
        st.metric("MAPE", f"{mape:.2f} %")

        st.subheader("Predicted vs Actual (test set — last 100 points)")
        comp_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
        comp_df = comp_df.reset_index(drop=True)
        fig4, ax4 = plt.subplots(1,1,figsize=(12,4))
        ax4.plot(comp_df['actual'].values[-200:], label='Actual')
        ax4.plot(comp_df['predicted'].values[-200:], label='Predicted', alpha=0.75)
        ax4.legend()
        ax4.set_ylabel("Global_active_power (kW)")
        st.pyplot(fig4)

        st.write("Download predictions as CSV:")
        csv = comp_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions.csv", csv, "predictions.csv", "text/csv")

# ---------- Task 3: Unsupervised Learning ----------
st.header("Task 3 — Anomaly Detection & Daily Clustering")

if st.button("Run Anomaly Detection & Clustering"):
    with st.spinner("Preparing hourly series for anomaly detection..."):
        hourly = hourly_resample(df_raw)['Global_active_power'].dropna()
        hr_df = pd.DataFrame({'power': hourly})
        hr_df['timestamp'] = hr_df.index
        hr_df = hr_df.reset_index(drop=True)

    with st.spinner("Running IsolationForest for anomalies..."):
        iso = IsolationForest(contamination=0.01, random_state=42)  # tune as needed
        hr_df['anomaly'] = iso.fit_predict(hr_df[['power']])
        hr_df['anomaly_flag'] = hr_df['anomaly'].apply(lambda x: 'anomaly' if x == -1 else 'normal')
        anomalies = hr_df[hr_df['anomaly_flag'] == 'anomaly']
        st.write(f"Detected anomalies (hourly): {len(anomalies)}")
        st.dataframe(anomalies.head(50))

        fig5, ax5 = plt.subplots(1,1,figsize=(12,4))
        ax5.plot(hr_df['timestamp'], hr_df['power'], label='power', alpha=0.7)
        ax5.scatter(anomalies['timestamp'], anomalies['power'], color='red', s=20, label='anomaly')
        ax5.set_ylabel("Global_active_power (kW)")
        ax5.legend()
        st.pyplot(fig5)

    with st.spinner("Building daily profiles for clustering..."):
        # Build daily profiles: pivot each day into 24-hour vector
        daily = hourly_resample(df_raw)['Global_active_power'].dropna().resample('D').apply(
            lambda g: g.resample('H').mean().values if len(g) > 0 else np.full(24, np.nan))
        # daily is a series of arrays; convert to DataFrame
        daily_profiles = []
        dates = []
        for d in hourly.resample('D'):
            dt = d[0]
            block = d[1]['Global_active_power'].resample('H').mean().values
            if len(block) == 24:
                daily_profiles.append(block)
                dates.append(dt)
        daily_profiles = np.array(daily_profiles)
        st.write("Daily profiles shape:", daily_profiles.shape)

        # handle rows with NaN by interpolation or dropping
        mask = ~np.isnan(daily_profiles).any(axis=1)
        daily_profiles_clean = daily_profiles[mask]
        dates_clean = np.array(dates)[mask]

        # scale before clustering
        scaler = StandardScaler()
        X_daily = scaler.fit_transform(daily_profiles_clean)

        # choose number of clusters (let user choose)
        n_clusters = st.slider("Select number of clusters for daily profiles", min_value=2, max_value=6, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_daily)

        st.write("Cluster counts:")
        counts = pd.Series(labels).value_counts().sort_index()
        st.dataframe(counts.rename("count"))

        # show cluster centers (inverse transform)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        fig6, ax6 = plt.subplots(figsize=(10,5))
        hours = np.arange(24)
        for i, center in enumerate(centers):
            ax6.plot(hours, center, label=f'Cluster {i}')
        ax6.set_xlabel("Hour of day")
        ax6.set_ylabel("Average Global_active_power (kW)")
        ax6.set_title("Cluster centers (daily consumption profiles)")
        ax6.legend()
        st.pyplot(fig6)

        # attach cluster label to dates and show one sample day per cluster
        sample_info = []
        for k in range(n_clusters):
            idxs = np.where(labels == k)[0]
            sample_day = dates_clean[idxs[0]] if len(idxs) > 0 else None
            avg_power = daily_profiles_clean[idxs].mean() if len(idxs) > 0 else np.nan
            sample_info.append({'cluster': k, 'sample_day': sample_day, 'avg_daily_power': float(np.nanmean(daily_profiles_clean[idxs])) if len(idxs) > 0 else np.nan, 'count': len(idxs)})
        st.subheader("Cluster characteristics (one row per cluster)")
        st.dataframe(pd.DataFrame(sample_info))

# ---------- Task 4: Simple Rule-Based AI ----------
st.header("Task 4 — Consumption Category Generator (rule-based)")

st.write("Based on a _predicted_ Global_active_power value (kW), assign a category and suggestion.")

pred_val = st.number_input("Enter predicted Global_active_power (kW):", min_value=0.0, format="%.3f", value=1.25)

def categorize_power(val):
    # Category thresholds chosen heuristically; adjust as needed
    if val < 1.5:
        return "Low Usage", "Good — keep up the efficient usage. Continue monitoring and maintain energy-saving habits."
    elif 1.5 <= val < 3.0:
        return "Medium Usage", "Moderate usage — consider running heavy appliances during off-peak hours and check appliance efficiency."
    else:
        return "High Usage", "High usage — reduce simultaneous heavy loads, inspect appliances for faults, and consider energy audits."

if st.button("Generate Category & Suggestion"):
    cat, sug = categorize_power(pred_val)
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:10px;background:#f7f9fb;">
        <h3>⚡ Usage Category: <b>{cat}</b></h3>
        <p><b>Suggestion:</b> {sug}</p>
        </div>
        """, unsafe_allow_html=True)
    st.write("### Example output")
    st.code(f"Predicted Power: {pred_val:.3f} kW  →  Category: {cat}  →  Suggestion: {sug}")

# ---------- Footer ----------
st.markdown("---")
st.write("Notes:")
st.write("""
- The forecasting here uses a simple RandomForest on lag-features (previous 24 hours). For production/time-series specialists, consider ARIMA/SARIMAX, Prophet, or deep learning models (LSTM/TCN/Transformer).
- Clustering uses KMeans on daily 24-hour profiles; interpret clusters by inspecting center curves and cluster sample days.
- Anomaly detection uses IsolationForest. Tune contamination and features to your needs.
""")
