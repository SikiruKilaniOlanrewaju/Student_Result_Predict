import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

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

    st.header("👨‍💻 About the Developer")
    st.markdown("""
    <div style="padding: 10px; background-color: #f1f1f1; border-radius: 10px;">
        <h4 style="color:#2c3e50;">Kilani Sikiru Olanrewaju</h4>
        <p style="font-size: 14px; line-height: 1.5;">
            🎓 <strong>Undergraduate Student</strong><br>
            Department of Computer Science,<br>
            Federal University of Agriculture, Abeokuta, Nigeria.
        </p>
        <p style="font-size: 14px; line-height: 1.5;">
            💡 <strong>Expertise:</strong><br>
            • Machine Learning & Data Analysis<br>
            • Full Stack Web Development<br>
            • Cybersecurity & IT Support<br>
        </p>
        <p style="font-size: 14px; line-height: 1.5;">
            🔧 <strong>Skills:</strong><br>
            Python, TensorFlow, Scikit-Learn, PHP, JavaScript, HTML/CSS, SQL, MySQL, Streamlit
        </p>
        <p style="font-size: 14px; line-height: 1.5;">
            📚 <strong>Certifications:</strong><br>
            • Cisco IT Security<br>
            • Cyber Threat Intelligence<br>
            • Soft Skills & Communication (Jobberman)
        </p>
        <p style="font-size: 14px; line-height: 1.5;">
            📫 <strong>Contact:</strong><br>
            <strong>Email:</strong> kilanisikiruolanrewaju@gmail.com<br>
            <strong>Phone:</strong> +234 806 152 7690
        </p>
        <p style="font-size: 13px; color: grey;">"Empowering solutions through code and creativity."</p>
    </div>
    """)

# Input Section
st.subheader("📥 Input WAEC Grades and GPA")
grades = {}
subjects = ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics"]

col1, col2 = st.columns(2)
for idx, subj in enumerate(subjects):
    with col1 if idx % 2 == 0 else col2:
        grades[subj] = st.selectbox(f"{subj} Grade", options=list(waec_mapping.keys()), key=subj)

gpa = st.number_input("🎯 Enter First Semester GPA", min_value=0.0, max_value=5.0, step=0.1)

# Prediction Function
def predict_result(input_data):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = result_mapping[predicted_class]
    confidence = np.max(prediction)
    return result, confidence

# Predict Button
if st.button("🔍 Predict Final Result"):
    input_data = [waec_mapping[grades[subj]] for subj in subjects]
    input_data.append(gpa)
    result, confidence = predict_result(input_data)
    
    st.success(f"✅ **Predicted Final Result: {result}**")
    st.write(f"📊 **Confidence Score: {confidence * 100:.2f}%**")

# CSV Upload for Batch Prediction
st.subheader("📤 Batch Prediction - Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if all(col in df.columns for col in ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics", "GPA"]):
            st.write("Input Data Preview:")
            st.write(df.head())

            predictions = []
            for _, row in df.iterrows():
                grades_input = [waec_mapping[row[subj]] for subj in subjects]
                grades_input.append(row['GPA'])
                result, confidence = predict_result(grades_input)
                predictions.append([result, confidence])

            df['Predicted Result'] = [pred[0] for pred in predictions]
            df['Confidence Score'] = [pred[1] for pred in predictions]

            st.write("Prediction Results:")
            st.write(df)

            # Download Button
            st.download_button(
                label="Download Prediction Results",
                data=df.to_csv(index=False),
                file_name="predicted_results.csv",
                mime="text/csv"
            )

            # Charts Section
            st.subheader("📊 Distribution of Predicted Final Results")
            result_counts = df['Predicted Result'].value_counts().reset_index()
            result_counts.columns = ['Final Result', 'Count']
            bar_chart = alt.Chart(result_counts).mark_bar().encode(
                x=alt.X('Final Result', sort=None),
                y='Count',
                color='Final Result'
            ).properties(width=600)
            st.altair_chart(bar_chart)

            st.subheader("📈 Confidence Score Histogram")
            fig, ax = plt.subplots()
            ax.hist(df['Confidence Score'], bins=10, color='skyblue', edgecolor='black')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of Predictions')
            st.pyplot(fig)

            st.subheader("📚 Average GPA per Final Result")
            avg_gpa = df.groupby('Predicted Result')['GPA'].mean().reset_index()
            gpa_chart = alt.Chart(avg_gpa).mark_bar().encode(
                x='Predicted Result',
                y='GPA',
                color='Predicted Result'
            ).properties(width=600)
            st.altair_chart(gpa_chart)

        else:
            st.error("The CSV file does not contain the necessary columns.")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.write("---")
st.markdown("<center>© 2025 Kilani Sikiru | Student Result Predictor App</center>", unsafe_allow_html=True)
