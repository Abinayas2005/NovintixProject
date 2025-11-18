# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Wind Turbine SCADA — Analytics", layout="wide", page_icon="🌬️")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_csv(uploaded):
    # Try to read CSV robustly
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=None, engine='python')
    return df

def parse_datetime(df, datetime_col=None):
    df = df.copy()
    # Try common datetime column names
    if datetime_col and datetime_col in df.columns:
        df['Datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
    else:
        # find likely datetime column
        candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if len(candidates) == 1:
            df['Datetime'] = pd.to_datetime(df[candidates[0]], errors='coerce')
        else:
            # if separate Date and Time columns exist
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce', dayfirst=False)
            else:
                # fallback: try first column
                df['Datetime'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df

def to_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def hourly_resample(df, freq):
    return df.resample(freq).mean()

def make_lag_features(series: pd.Series, window:int):
    df = pd.DataFrame({'y': series})
    for i in range(1, window+1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['target'] = df['y'].shift(-1)
    df = df.dropna()
    X = df[[f'lag_{i}' for i in range(1, window+1)]]
    y = df['target']
    return X, y

def train_test_time_split(X, y, test_frac=0.2):
    split = int(len(X)*(1-test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    return mae, rmse, mape

# -------------------------
# Sidebar - Navigation / Inputs
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Forecasting", "Anomaly Detection", "Performance Score"])

st.sidebar.markdown("### Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV (Wind Turbine SCADA)", type=["csv","txt","zip"])
use_sample = st.sidebar.checkbox("Use small synthetic sample (demo)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Forecast settings")
resample_freq = st.sidebar.selectbox("Resample frequency (for forecasting/EDA)", options=['H','30T','15T','D'], index=0,
                                     format_func=lambda x: {'H':'Hourly','30T':'30-min','15T':'15-min','D':'Daily'}[x])
window_size = st.sidebar.number_input("Window size (lags) for forecasting", min_value=1, max_value=168, value=24)
test_frac = st.sidebar.slider("Test fraction (time-series split)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
contamination = st.sidebar.slider("Anomaly contamination (IsolationForest)", min_value=0.0, max_value=0.2, value=0.02, step=0.01)

# -------------------------
# Load data
# -------------------------
if use_sample and uploaded is None:
    st.info("Using synthetic demo dataset (small).")
    rng = pd.date_range("2022-01-01", periods=24*90, freq='H')  # 90 days hourly
    rng = rng.tz_localize(None)
    np.random.seed(42)
    df = pd.DataFrame({
        'Datetime': rng,
        'LV ActivePower (kW)': 1000 * (0.5 + 0.5*np.sin(np.linspace(0,10,len(rng))) ) + np.random.randn(len(rng))*50,
        'Wind Speed (m/s)': 8 + 3*np.sin(np.linspace(0,5,len(rng))) + np.random.randn(len(rng))*0.8,
        'Theoretical_Power_Curve (kWh)': 1000 * (0.4 + 0.6*np.clip(np.sin(np.linspace(0,10,len(rng))),0,1)),
        'Wind Direction (°)': np.mod(180 + 30*np.sin(np.linspace(0,8,len(rng))) + np.random.randn(len(rng))*5, 360)
    }).set_index('Datetime')
else:
    if uploaded is None:
        st.info("Upload the Kaggle CSV (or enable sample).")
        st.stop()
    else:
        with st.spinner("Loading CSV..."):
            raw = load_csv(uploaded)
            # Try to find correct column names in dataset (variable names may vary)
            # Provide suggested mapping by user later
            df = raw.copy()

# Allow user to map columns if dataset uploaded
if 'df' in locals() and uploaded is not None:
    st.sidebar.markdown("### Column mapping (if auto-detection fails)")
    cols = df.columns.tolist()
    dt_col = st.sidebar.selectbox("Datetime column (if present)", options=[None]+cols, index=0)
    pwr_col = st.sidebar.selectbox("LV ActivePower column", options=cols, index=cols.index(next((c for c in cols if 'active' in c.lower() or 'power' in c.lower()), cols[0])))
    ws_col = st.sidebar.selectbox("Wind Speed column", options=cols, index=cols.index(next((c for c in cols if 'wind' in c.lower() and 'speed' in c.lower()), cols[0])))
    theo_col = st.sidebar.selectbox("Theoretical Power Curve column", options=cols, index=cols.index(next((c for c in cols if 'theoretical' in c.lower()), cols[0])))
    wd_col = st.sidebar.selectbox("Wind Direction column", options=cols, index=cols.index(next((c for c in cols if 'direction' in c.lower()), cols[0])))
    # parse datetime / rename columns
    df = parse_datetime(df, datetime_col=dt_col)
    df = to_numeric_cols(df, [pwr_col, ws_col, theo_col, wd_col])
    # rename to canonical names used in app
    df = df.rename(columns={
        pwr_col: 'LV ActivePower (kW)',
        ws_col: 'Wind Speed (m/s)',
        theo_col: 'Theoretical_Power_Curve (kWh)',
        wd_col: 'Wind Direction (°)'
    })
    # keep canonical columns only
    df = df[['LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (kWh)','Wind Direction (°)']]

# At this point df should exist and be a DataFrame indexed by Datetime
st.title("🌬️ Wind Turbine SCADA — Single-file App")
st.markdown("Tasks: EDA, Forecasting (all 4 variables), Anomaly Detection (underperformance), Turbine Performance Score (AI rule).")

# -------------------------
# Page: Home
# -------------------------
if page == "Home":
    st.header("Overview & Instructions")
    st.markdown("""
    **How to use**
    1. Upload the CSV containing Date/Time and the required columns, or use the small synthetic demo.  
    2. Use **EDA** to inspect time-series and missing/abnormal values.  
    3. Use **Forecasting** to train simple RandomForest models (windowed lags) for each variable.  
    4. Use **Anomaly Detection** to find underperformance vs theoretical power.  
    5. Use **Performance Score** to compute a 0–100 performance score and get recommendations.
    """)
    st.subheader("Data preview")
    st.dataframe(df.head(10))

# -------------------------
# Page: EDA
# -------------------------
elif page == "EDA":
    st.header("Task 1 — EDA")
    st.subheader("Time-series trend for all variables (resampled view)")

    freq = resample_freq
    df_r = hourly_resample(df, freq)
    st.write(f"Resampled to {freq}. Rows: {len(df_r)}")
    cols = df_r.columns.tolist()

    # Plot each series
    fig, axes = plt.subplots(2,2, figsize=(14,8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.lineplot(x=df_r.index, y=df_r[col], ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missing & Abnormal readings")
    missing = df_r.isna().sum()
    st.dataframe(missing.rename("missing_count"))

    # Abnormal if negative (power) or outside 0-360 deg for wind direction; or > 99.5 quantile
    abnormalities = {}
    if 'LV ActivePower (kW)' in df_r:
        pwr = df_r['LV ActivePower (kW)']
        high_cut = pwr.quantile(0.995)
        abnormalities['LV ActivePower (kW) - neg'] = int((pwr < 0).sum())
        abnormalities['LV ActivePower (kW) - >99.5%'] = int((pwr > high_cut).sum())
    if 'Wind Speed (m/s)' in df_r:
        ws = df_r['Wind Speed (m/s)']
        abnormalities['Wind Speed - neg'] = int((ws < 0).sum())
    if 'Wind Direction (°)' in df_r:
        wd = df_r['Wind Direction (°)']
        abnormalities['Wind Direction - out_of_range'] = int((~wd.between(0,360)).sum())
    st.write("Abnormal counts (heuristic):")
    st.table(pd.Series(abnormalities, name="count"))

    st.subheader("Wind Speed vs LV ActivePower — Scatter (power curve shape)")
    if df_r['Wind Speed (m/s)'].notna().sum() > 0 and df_r['LV ActivePower (kW)'].notna().sum() > 0:
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=df_r['Wind Speed (m/s)'], y=df_r['LV ActivePower (kW)'], alpha=0.4, s=10, ax=ax2)
        ax2.set_xlabel("Wind Speed (m/s)")
        ax2.set_ylabel("LV ActivePower (kW)")
        ax2.set_title("Power curve (scatter)")
        st.pyplot(fig2)
    else:
        st.info("Not enough data for scatter plot.")

# -------------------------
# Page: Forecasting
# -------------------------
elif page == "Forecasting":
    st.header("Task 2 — Time-Series Forecasting (all 4 variables)")

    freq = resample_freq
    df_r = hourly_resample(df, freq).dropna(how='all')  # if all NaN drop row
    st.write(f"Resampled to {freq}. Rows: {len(df_r)}")

    targets = ['LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (kWh)','Wind Direction (°)']
    available_targets = [t for t in targets if t in df_r.columns and df_r[t].notna().sum() > window_size+5]
    st.write("Targets available for forecasting:", available_targets)

    if not available_targets:
        st.error("No target has enough data for windowed forecasting. Reduce window or resample differently.")
    else:
        run_forecast = st.button("Run forecasting for all available targets")
        if run_forecast:
            results = {}
            for tgt in available_targets:
                st.subheader(f"Forecasting: {tgt}")
                series = df_r[tgt].interpolate(limit_direction='both')
                X, y = make_lag_features(series, window_size)
                X_train, X_test, y_train, y_test = train_test_time_split(X, y, test_frac=test_frac)
                st.write(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae, rmse, mape = compute_metrics(y_test, y_pred)
                st.write(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}  MAPE: {mape:.2f}%")

                # plot predicted vs actual (last N)
                comp = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
                comp = comp.reset_index(drop=True)
                N = min(200, len(comp))
                fig3, ax3 = plt.subplots(figsize=(10,4))
                ax3.plot(comp['actual'].values[-N:], label='Actual')
                ax3.plot(comp['predicted'].values[-N:], label='Predicted', alpha=0.8)
                ax3.set_title(f"{tgt} - Predicted vs Actual (last {N} points)")
                ax3.legend()
                st.pyplot(fig3)

                # store
                results[tgt] = {'y_test': y_test, 'y_pred': y_pred, 'mae':mae,'rmse':rmse,'mape':mape, 'comp_df':comp}

            st.success("Forecasting completed for available targets.")
            # Offer download of predictions for all targets combined (aligned by test index)
            # We'll align by index of smallest common test length (simple)
            combined = []
            for t, r in results.items():
                tmp = pd.DataFrame({'target': t, 'actual': r['y_test'].values, 'predicted': r['y_pred']})
                combined.append(tmp.reset_index(drop=True))
            out_df = pd.concat(combined, axis=0).reset_index(drop=True)
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download all predictions CSV", csv, "predictions_all.csv", "text/csv")

# -------------------------
# Page: Anomaly Detection
# -------------------------
elif page == "Anomaly Detection":
    st.header("Task 3 — Anomaly Detection (underperformance vs theoretical)")

    df_r = hourly_resample(df, resample_freq)
    if 'LV ActivePower (kW)' not in df_r.columns or 'Theoretical_Power_Curve (kWh)' not in df_r.columns:
        st.error("Need both LV ActivePower and Theoretical_Power_Curve columns for underperformance detection.")
    else:
        df_r = df_r.dropna(subset=['LV ActivePower (kW)','Theoretical_Power_Curve (kWh)'])
        df_r['residual'] = df_r['Theoretical_Power_Curve (kWh)'] - df_r['LV ActivePower (kW)']
        # underperformance when residual is positive and large (theoretical >> actual)
        st.write("Residual = Theoretical - Actual. Positive large residual = underperformance.")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_r.index, df_r['residual'], label='residual')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("Residual (kW)")
        ax.set_title("Residual (Theoretical - Actual) over time")
        st.pyplot(fig)

        # Use IsolationForest on residuals to flag anomalies
        iso = IsolationForest(contamination=contamination, random_state=42)
        X_iso = df_r[['residual']].fillna(0)
        df_r['anomaly'] = iso.fit_predict(X_iso)
        df_r['anomaly_flag'] = df_r['anomaly'].map({1:'normal', -1:'anomaly'})

        anomalies = df_r[df_r['anomaly'] == -1].sort_index()
        st.write("Anomalies detected (underperformance points):", len(anomalies))
        st.dataframe(anomalies[['LV ActivePower (kW)','Theoretical_Power_Curve (kWh)','residual']].head(50))

        # show chart with anomalies highlighted
        fig2, ax2 = plt.subplots(figsize=(12,4))
        ax2.plot(df_r.index, df_r['residual'], label='residual', alpha=0.7)
        ax2.scatter(anomalies.index, anomalies['residual'], color='red', s=15, label='anomaly')
        ax2.legend()
        ax2.set_title("Underperformance anomalies highlighted")
        st.pyplot(fig2)

        st.info("Interpretation: Points with large positive residual where actual << theoretical suggest underperformance (mechanical issue, curtailment, sensor error). Review timestamps and operational logs.")

# -------------------------
# Page: Performance Score (AI Task)
# -------------------------
elif page == "Performance Score":
    st.header("Task 4 — AI Turbine Performance Score Generator")

    df_r = hourly_resample(df, resample_freq)
    # require both columns
    if 'LV ActivePower (kW)' not in df_r.columns or 'Theoretical_Power_Curve (kWh)' not in df_r.columns:
        st.error("Need both LV ActivePower and Theoretical_Power_Curve columns.")
    else:
        df_score = df_r.dropna(subset=['LV ActivePower (kW)','Theoretical_Power_Curve (kWh)']).copy()
        # performance ratio actual/theoretical, clip to [0, inf)
        df_score['ratio'] = df_score['LV ActivePower (kW)'] / df_score['Theoretical_Power_Curve (kWh)']
        df_score['ratio'] = df_score['ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        # scale ratio to 0-100 by clipping ratio to [0,1] then *100; if theoretical is 0, ratio becomes 0
        df_score['score_raw'] = df_score['ratio'].clip(lower=0, upper=1)
        df_score['score'] = (df_score['score_raw'] * 100).round(1)

        # categorize
        def categorize(s):
            if s >= 80:
                return 'Good'
            elif s >= 50:
                return 'Moderate'
            else:
                return 'Poor'
        df_score['category'] = df_score['score'].apply(categorize)

        st.subheader("Score distribution (resampled rows)")
        st.write("Rows used:", len(df_score))
        st.write(df_score['category'].value_counts())

        fig, ax = plt.subplots(figsize=(10,4))
        sns.histplot(df_score['score'], bins=30, ax=ax)
        ax.set_xlabel("Performance Score (0-100)")
        st.pyplot(fig)

        st.subheader("Example outputs (first 10 rows)")
        out = df_score[['LV ActivePower (kW)','Theoretical_Power_Curve (kWh)','score','category']].head(10)
        st.dataframe(out)

        st.subheader("Automated suggestion rules")
        st.markdown("""
        - **Good (score ≥ 80):** Turbine performing near expectations. Routine monitoring recommended.  
        - **Moderate (50 ≤ score < 80):** Reduced performance — inspect for minor issues, check blade cleaning and pitch control.  
        - **Poor (score < 50):** Significant underperformance — schedule immediate mechanical/electrical inspection, check yaw/pitch and generator.
        """)

        st.subheader("Generate suggestion for a chosen timestamp")
        chosen_idx = st.selectbox("Choose timestamp (index) for suggestion", options=df_score.index[:500] if len(df_score)>500 else df_score.index)
        if chosen_idx is not None:
            row = df_score.loc[chosen_idx]
            score = row['score']
            cat = row['category']
            suggestion_map = {
                'Good': "Performance is good. Continue routine monitoring and logging.",
                'Moderate': "Moderate performance — inspect blade condition, clean if needed, check pitch/yaw calibration.",
                'Poor': "Poor performance — prioritize maintenance: inspect generator, gearbox, pitch and yaw systems, and check SCADA alarms."
            }
            st.markdown(f"""
            **Timestamp:** {chosen_idx}  
            **Actual Power:** {row['LV ActivePower (kW)']:.2f} kW  
            **Theoretical Power:** {row['Theoretical_Power_Curve (kWh)']:.2f} kW  
            **Performance score:** {score}  
            **Category:** {cat}  

            **Suggestion:** {suggestion_map[cat]}
            """)

        st.success("Performance scoring complete.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("Notes: This app uses simple, interpretable models for demonstration (RandomForest with lag features). For production forecasting consider time-series specific models (SARIMA, Prophet, LSTM/Transformer). Tuning thresholds (score bins, anomaly contamination) is recommended for your turbine and site.")
