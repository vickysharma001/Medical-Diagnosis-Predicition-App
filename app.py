import joblib
import pandas as pd
import streamlit as st

# Sabhi saved files load karna
lr_model = joblib.load("logistic_model.pkl")
dt_model = joblib.load("dt_model.pkl")  # Decision Tree load kiya
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🏥 Medical Diagnosis Prediction App")
st.write(
    "Patient ke symptoms daaliye aur choose kijiye ki aap kaunse model se"
    " prediction chahte hain."
)

# Model select karne ka option
model_choice = st.selectbox(
    "Choose Machine Learning Model", ["Logistic Regression", "Decision Tree"]
)

# Patient Details Input
st.subheader("Patient Details:")

fever = st.selectbox("Fever (0 = No, 1 = Yes)", [0, 1])
cough = st.selectbox("Cough (0 = No, 1 = Yes)", [0, 1])
chest_pain = st.selectbox("Chest Pain (0 = No, 1 = Yes)", [0, 1])
breathlessness = st.selectbox("Breathlessness (0 = No, 1 = Yes)", [0, 1])
temperature = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

# Prediction Button
if st.button("Predict Diagnosis"):
  input_data = pd.DataFrame(
      [[fever, cough, chest_pain, breathlessness, temperature]],
      columns=["fever", "cough", "chest_pain", "breathlessness", "temperature"],
  )

  # Model ke hisab se prediction karna
  if model_choice == "Logistic Regression":
    # Logistic Regression ke liye scaling zaroori hai
    scaled_data = scaler.transform(input_data)
    prediction_idx = lr_model.predict(scaled_data)[0]
    probabilities = lr_model.predict_proba(scaled_data)[0]
  else:
    # Decision Tree ko scaling ki zaroorat nahi hoti
    prediction_idx = dt_model.predict(input_data)[0]
    probabilities = dt_model.predict_proba(input_data)[0]

  predicted_disease = le.inverse_transform([prediction_idx])[0]

  # Result display karna
  st.success(
      f"**Model Used:** {model_choice} \n\n **Predicted Diagnosis:**"
      f" {predicted_disease}"
  )

  # Probabilities breakdown
  st.write("### Probability Breakdown:")
  prob_df = pd.DataFrame(
      {"Disease": le.classes_, "Probability": probabilities}
  ).set_index("Disease")

  st.bar_chart(prob_df)