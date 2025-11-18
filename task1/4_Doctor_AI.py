import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.title("🧠 AI Doctor Recommendation System")

uploaded_file = st.file_uploader("Upload Healthcare Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    encoder = LabelEncoder()
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object"):
        df_enc[col] = encoder.fit_transform(df_enc[col])

    X = df_enc.drop("Test Results", axis=1)
    y = df_enc["Test Results"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    idx = X_test.sample(1).index[0]
    prediction = model.predict([X_test.loc[idx]])[0]

    age = df.loc[idx, "Age"]
    condition = df.loc[idx, "Medical Condition"]
    medication = df.loc[idx, "Medication"]

    st.success("AI Recommendation Generated!")

    st.markdown(f"""
    ### 🩺 AI Doctor Recommendation

    **Patient Age:** {age}  
    **Medical Condition:** {condition}  
    **Medication:** {medication}  
    **Predicted Test Result:** {prediction:.2f}

    Based on these findings, the patient is advised to maintain their medication routine
    and follow proper rest and hydration. If symptoms worsen, a check-up is recommended.

    — *AI Medical Assistant*
    """)
else:
    st.info("Upload your dataset to generate an AI recommendation.")
