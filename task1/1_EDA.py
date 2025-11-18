import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload Healthcare Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    tab1, tab2 = st.tabs(["📈 Numerical Distributions", "📦 Categorical Frequency"])

    with tab1:
        st.subheader("Distribution of Age, Billing Amount, Room Number")
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(df["Age"], kde=True, ax=ax[0])
        sns.histplot(df["Billing Amount"], kde=True, ax=ax[1])
        sns.histplot(df["Room Number"], kde=True, ax=ax[2])
        st.pyplot(fig)

    with tab2:
        st.subheader("Categorical Feature Frequencies")
        fig, ax = plt.subplots(1, 3, figsize=(22, 5))
        sns.countplot(x=df["Medical Condition"], ax=ax[0])
        sns.countplot(x=df["Admission Type"], ax=ax[1])
        sns.countplot(x=df["Medication"], ax=ax[2])
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")
