import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('student_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

waec_mapping = {'A1': 6, 'B2': 5, 'B3': 4, 'C4': 3, 'C5': 2, 'C6': 1, 'D7': 0, 'F9': 0}
result_mapping = {0: 'First Class', 1: 'Second Upper', 2: 'Second Lower', 3: 'Third Class', 4: 'Pass/Withdrawn'}

# Streamlit Title
st.title("🎓 Student Final Result Prediction - DEVELOPED BY KILANI SIKIRU O - PT/22/0061")

# User input for grades and GPA
grades = {}
subjects = ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics"]
for subj in subjects:
    grades[subj] = st.selectbox(f"{subj} Grade", list(waec_mapping.keys()))

gpa = st.number_input("First Semester GPA", min_value=0.0, max_value=5.0, step=0.1)

# Function to predict for individual input
def predict_result(input_data):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = result_mapping[predicted_class]
    confidence = np.max(prediction)  # Get the highest confidence score
    return result, confidence

# Prediction Button
if st.button("Predict"):
    input_data = [waec_mapping[grades[subj]] for subj in subjects]
    input_data.append(gpa)
    result, confidence = predict_result(input_data)
    st.success(f"✅ Predicted Final Result: **{result}**")
    st.write(f"📊 Confidence Score: **{confidence*100:.2f}%**")

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
