import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit page setup
st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide")

# App Title
st.title("🧬 Lung Cancer Risk Prediction & Analysis Web App")
st.markdown("Upload your dataset, explore data, train a model, and predict lung cancer risk (single or batch).")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Lung Cancer CSV Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
    df["GENDER"] = LabelEncoder().fit_transform(df["GENDER"])

    # ----- EDA Section -----
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    st.bar_chart(df["LUNG_CANCER"].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ----- Model Training -----
    st.header("🧠 Model Training")
    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "lung_cancer_model.pkl")
    st.success("✅ Model trained and saved as lung_cancer_model.pkl")

    # ----- Single Prediction -----
    st.header("🧾 Single Entry Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["M", "F"])
        age = col2.slider("Age", 15, 100, 45)
        smoking = col1.selectbox("Smoking", [1, 0])
        yellow_fingers = col2.selectbox("Yellow Fingers", [1, 0])
        anxiety = col1.selectbox("Anxiety", [1, 0])
        peer_pressure = col2.selectbox("Peer Pressure", [1, 0])
        chronic = col1.selectbox("Chronic Disease", [1, 0])
        fatigue = col2.selectbox("Fatigue", [1, 0])
        allergy = col1.selectbox("Allergy", [1, 0])
        wheezing = col2.selectbox("Wheezing", [1, 0])
        alcohol = col1.selectbox("Alcohol Consuming", [1, 0])
        coughing = col2.selectbox("Coughing", [1, 0])
        shortness = col1.selectbox("Shortness of Breath", [1, 0])
        swallowing = col2.selectbox("Swallowing Difficulty", [1, 0])
        chest_pain = col1.selectbox("Chest Pain", [1, 0])
        predict_btn = st.form_submit_button("Predict")

    if predict_btn:
        input_df = pd.DataFrame([{
            "GENDER": 1 if gender == "M" else 0,
            "AGE": age,
            "SMOKING": smoking,
            "YELLOW_FINGERS": yellow_fingers,
            "ANXIETY": anxiety,
            "PEER_PRESSURE": peer_pressure,
            "CHRONIC DISEASE": chronic,
            "FATIGUE ": fatigue,
            "ALLERGY ": allergy,
            "WHEEZING": wheezing,
            "ALCOHOL_CONSUMING": alcohol,
            "COUGHING": coughing,
            "SHORTNESS OF BREATH": shortness,
            "SWALLOWING DIFFICULTY": swallowing,
            "CHEST PAIN": chest_pain
        }])
        model = joblib.load("lung_cancer_model.pkl")
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Lung Cancer Risk Prediction: {'YES' if prediction == 1 else 'NO'}")

    # ----- Batch Prediction -----
    st.header("📁 Batch Prediction (Upload CSV)")
    batch_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv", key="batch")

    if batch_file:
        batch_df = pd.read_csv(batch_file)
        batch_df["GENDER"] = LabelEncoder().fit_transform(batch_df["GENDER"])
        model = joblib.load("lung_cancer_model.pkl")
        batch_preds = model.predict(batch_df)
        batch_df["PREDICTED_LUNG_CANCER"] = ["YES" if p == 1 else "NO" for p in batch_preds]
        st.write(batch_df)
        st.download_button("Download Predictions", batch_df.to_csv(index=False), file_name="predicted_lung_cancer.csv")

else:
    st.info("⬆️ Please upload a lung cancer dataset to begin.")
