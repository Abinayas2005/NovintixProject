import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("🤖 Test Results Prediction (Supervised Learning)")

uploaded_file = st.file_uploader("Upload Healthcare Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_ml = df.copy()
    
    encoder = LabelEncoder()
    for col in df_ml.select_dtypes(include="object"):
        df_ml[col] = encoder.fit_transform(df_ml[col])

    X = df_ml.drop("Test Results", axis=1)
    y = df_ml["Test Results"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Model Performance")
    st.write(f"📌 R² Score: {model.score(X_test, y_test):.4f}")

    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Test Results")
    st.pyplot(fig)
else:
    st.info("Upload your dataset to start model training.")
