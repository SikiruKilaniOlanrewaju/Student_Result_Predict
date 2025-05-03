import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('student_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

waec_mapping = {'A1': 6, 'B2': 5, 'B3': 4, 'C4': 3, 'C5': 2, 'C6': 1, 'D7': 0, 'F9': 0}
result_mapping = {0: 'First Class', 1: 'Second Upper', 2: 'Second Lower', 3: 'Third Class', 4: 'Pass/Withdrawn'}

st.title("🎓 Student Final Result Prediction - DEVELOPED BY KILANI SIKIRU O - PT/22/0061")

grades = {}
subjects = ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics"]
for subj in subjects:
    grades[subj] = st.selectbox(f"{subj} Grade", list(waec_mapping.keys()))

gpa = st.number_input("First Semester GPA", min_value=0.0, max_value=5.0, step=0.1)

if st.button("Predict"):
    input_data = [waec_mapping[grades[subj]] for subj in subjects]
    input_data.append(gpa)
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = result_mapping[predicted_class]
    st.success(f"✅ Predicted Final Result: **{result}**")
