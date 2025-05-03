import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('student_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

# Grade and result mappings
waec_mapping = {'A1': 6, 'B2': 5, 'B3': 4, 'C4': 3, 'C5': 2, 'C6': 1, 'D7': 0, 'F9': 0}
result_mapping = {
    0: '🎓 First Class',
    1: '🥈 Second Class Upper',
    2: '🥉 Second Class Lower',
    3: '🎗️ Third Class',
    4: '⚠️ Pass/Withdrawn'
}

# Page Configuration
st.set_page_config(page_title="Student Result Predictor", page_icon="🎓", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f9f4;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🎓 Student Final Result Prediction")
st.markdown("##### Developed by **Kilani Sikiru O.**  &nbsp;&nbsp;&nbsp; 🎓 Matric: **PT/22/0061**")
st.write("---")

# Sidebar Information
with st.sidebar:
    st.header("📘 About App")
    st.info("""
        This app uses a machine learning model to predict a student's **final academic result**
        based on their WAEC grades and first semester GPA.
        
        Inputs:
        - WAEC Grades for 6 subjects
        - First Semester GPA
        
        Output:
        - Predicted Final Degree Class
    """)

# Input Section
st.subheader("📥 Input WAEC Grades and GPA")
grades = {}
subjects = ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics"]

col1, col2 = st.columns(2)
for idx, subj in enumerate(subjects):
    with col1 if idx % 2 == 0 else col2:
        grades[subj] = st.selectbox(f"{subj} Grade", options=list(waec_mapping.keys()), key=subj)

gpa = st.slider("🎯 First Semester GPA", 0.0, 5.0, step=0.1)

# Predict Button
if st.button("🔍 Predict Final Result"):
    input_data = [waec_mapping[grades[subj]] for subj in subjects]
    input_data.append(gpa)
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = result_mapping[predicted_class]
    
    st.success(f"✅ **Predicted Final Result: {result}**")

# Footer
st.write("---")
st.markdown("<center>© 2025 Kilani Sikiru | Student Result Predictor App</center>", unsafe_allow_html=True)
