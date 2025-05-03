import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

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

# GPA as number input
gpa = st.number_input("🎯 Enter First Semester GPA", min_value=0.0, max_value=5.0, step=0.1)

# Function to predict for individual input
def predict_result(input_data):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = result_mapping[predicted_class]
    confidence = np.max(prediction)  # Get the highest confidence score
    return result, confidence

# Predict Button
if st.button("🔍 Predict Final Result"):
    input_data = [waec_mapping[grades[subj]] for subj in subjects]
    input_data.append(gpa)
    result, confidence = predict_result(input_data)
    
    st.success(f"✅ **Predicted Final Result: {result}**")
    st.write(f"📊 **Confidence Score: {confidence * 100:.2f}%**")

# **CSV Upload for Batch Prediction**
st.subheader("📤 Batch Prediction - Upload CSV")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Check if the necessary columns exist
        if all(col in df.columns for col in ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics", "GPA"]):
            st.write("Input Data Preview:")
            st.write(df.head())

            # Prepare data for prediction
            predictions = []
            for _, row in df.iterrows():
                grades_input = [waec_mapping[row[subj]] for subj in subjects]
                grades_input.append(row['GPA'])
                result, confidence = predict_result(grades_input)
                predictions.append([result, confidence])

            # Create a DataFrame with predictions
            df['Predicted Result'] = [pred[0] for pred in predictions]
            df['Confidence Score'] = [pred[1] for pred in predictions]

            # Display the results
            st.write("Prediction Results:")
            st.write(df)

            # Option to download the results as a CSV
            st.download_button(
                label="Download Prediction Results",
                data=df.to_csv(index=False),
                file_name="predicted_results.csv",
                mime="text/csv"
            )
        else:
            st.error("The CSV file does not contain the necessary columns.")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.write("---")
st.markdown("<center>© 2025 Kilani Sikiru | Student Result Predictor App</center>", unsafe_allow_html=True)
