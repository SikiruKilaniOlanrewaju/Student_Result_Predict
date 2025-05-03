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
    footer {
        font-size: 14px;
        text-align: center;
        color: gray;
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px 0;
        background-color: #f4f9f4;
        border-top: 1px solid #ddd;
    }
    footer a {
        color: #28a745;
        text-decoration: none;
    }
    footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Check if welcome screen has been passed
if 'welcome_screen' not in st.session_state:
    st.session_state.welcome_screen = True

# If we're on the welcome screen
if st.session_state.welcome_screen:
    # Welcome page content
    st.title("🎓 Welcome to the Student Result Predictor")
    st.write("This app predicts the final academic result of a student based on their WAEC grades and first semester GPA.")
    st.write("---")
    
    st.markdown("""
        <div style="text-align:center;">
            <h3 style="color:#2c3e50;">Get ready to predict your final result with just a few details!</h3>
            <p style="font-size:16px; color:#7f8c8d;">This application helps you forecast your academic future based on historical data and grades.</p>
            <p style="font-size:16px; color:#7f8c8d;">Click the button below to proceed and input your data.</p>
        </div>
    """, unsafe_allow_html=True)

    # Proceed button
    if st.button("🔜 Proceed to Prediction Page"):
        st.session_state.welcome_screen = False  # Move to the next page

# Main App Content (Prediction Page)
else:
    st.title("🎓 Student Final Result Prediction")
    st.markdown("##### Developed by **Kilani Sikiru O.**  &nbsp;&nbsp;&nbsp; 🎓 Matric: **PT/22/0061**")
    st.write("---")

    # Sidebar Information
    with st.sidebar:
        st.header("📘 About App")
        st.info("""This app uses a machine learning model to predict a student's **final academic result**
                  based on their WAEC grades and first semester GPA.""")
    
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
        """, unsafe_allow_html=True)

    # Input form for GPA and WAEC grades
    st.header("📊 Enter Your Details")
    with st.form("input_form"):
        # Collecting input data
        gpa = st.number_input("First Semester GPA", min_value=0.0, max_value=5.0, step=0.1)
        waec_grades = {
            subject: st.selectbox(f"Select grade for {subject}", options=list(waec_mapping.keys())) for subject in
            ["English", "Mathematics", "Physics", "Chemistry", "Biology"]
        }

        # Button to submit the form
        submit_button = st.form_submit_button("🔮 Predict Result")

        if submit_button:
            # Preprocess the input data
           # Preprocess the input data
waec_scores = np.array([waec_mapping[grade] for grade in waec_grades.values()])
features = np.concatenate([waec_scores, [gpa]])

# Ensure the features are reshaped correctly
features = features.reshape(1, -1)  # Ensure the shape is (1, n_features)

# Scale the features
scaled_features = scaler.transform(features)  # Now pass the reshaped array

# Predict the result
prediction = model.predict(scaled_features)
result_class = np.argmax(prediction)
result = result_mapping[result_class]

            st.success(f"Your predicted result is: {result}")

            # Show the prediction chart
            st.subheader("📊 Prediction Confidence")
            result_chart_data = pd.DataFrame(prediction[0], columns=["Confidence"], index=["First Class", "Second Class Upper", "Second Class Lower", "Third Class", "Pass/Withdrawn"])
            st.bar_chart(result_chart_data)

# Footer content
st.markdown("""
    <footer>
        <p>Powered by <a href="https://www.github.com/kilanisikiru" target="_blank">Kilani Sikiru Olanrewaju</a> | 
        <span style="color:#7f8c8d;">&copy; 2025</span></p>
    </footer>
""", unsafe_allow_html=True)
