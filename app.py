import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('student_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

# Mapping dictionaries
waec_mapping = {'A1': 6, 'B2': 5, 'B3': 4, 'C4': 3, 'C5': 2, 'C6': 1, 'D7': 0, 'F9': 0}
result_mapping = {
    0: '🎓 First Class',
    1: '🏅 Second Upper',
    2: '📘 Second Lower',
    3: '📙 Third Class',
    4: '❗ Pass/Withdrawn'
}

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f7fdfc;
        }
        h1 {
            color: #004d00;
            font-size: 36px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #198754;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.5em 1.5em;
            margin-top: 1em;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            font-size: small;
            color: grey;
            margin-top: 2em;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 About This App")
st.sidebar.info(
    "This app predicts a student's **likely final university result** based on WAEC grades "
    "and first semester GPA.\n\nDeveloped by **Kilani Sikiru O. (PT/22/0061)**"
)

# Main Title
st.title("🎓 Student Final Result Prediction")
st.markdown("### Developed by Kilani Sikiru O. – PT/22/0061")

# Input Section
st.subheader("📥 Enter Your WAEC Grades")
grades = {}
subjects = ["English", "Maths", "Physics", "Chemistry", "Biology", "Economics"]

col1, col2 = st.columns(2)
for i, subj in enumerate(subjects):
    with col1 if i % 2 == 0 else col2:
        grades[subj] = st.selectbox(f"{subj} Grade", list(waec_mapping.keys()))

st.subheader("📊 Enter First Semester GPA")
gpa = st.number_input("GPA (0.0 - 5.0)", min_value=0.0, max_value=5.0, step=0.1)

# Prediction
if st.button("🔮 Predict Final Result"):
    with st.spinner("Analyzing your data..."):
        input_data = [waec_mapping[grades[subj]] for subj in subjects]
        input_data.append(gpa)
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = result_mapping[predicted_class]

    st.success(f"✅ **Predicted Final Result:** {result}")

# Footer
st.markdown("<div class='footer'>© 2025 Kilani Sikiru O. | All Rights Reserved</div>", unsafe_allow_html=True)
