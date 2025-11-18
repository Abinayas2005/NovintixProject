import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚨 Billing Amount Anomaly Detection")

uploaded_file = st.file_uploader("Upload Healthcare Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[["Billing Amount"]])
    df["Anomaly_Flag"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    st.subheader("Detected Anomalies")
    anomalies = df[df["Anomaly_Flag"] == "Anomaly"]
    st.dataframe(anomalies)

    fig = plt.figure(figsize=(10,5))
    sns.scatterplot(
        x=np.arange(len(df)),
        y=df["Billing Amount"],
        hue=df["Anomaly_Flag"]
    )
    plt.title("Billing Amount Anomalies")
    st.pyplot(fig)

else:
    st.info("Upload your dataset to detect anomalies.")
