import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open(r'/workspaces/Heart-disease2/project/model_LogisticRegression.pkl','rb'))

# Function to predict heart disease with a custom threshold
def predict_heart_disease(features, threshold=0.17):
    # Get the probability for the positive class (class 1)
    prob = model.predict_proba([features])[0][1]
    
    # Apply the custom threshold
    if prob >= threshold:
        return 1  # Positive (Heart disease)
    else:
        return 0  # Negative (No heart disease)

# Title of the app
st.title('Heart Disease Prediction')

# Add input fields for each feature based on the provided list
male = st.selectbox('Gender (0 for female, 1 for male)', [0, 1])
age = st.slider('Age', min_value=18, max_value=100, value=50)
current_smoker = st.selectbox('Current Smoker (0 for No, 1 for Yes)', [0, 1])
cigs_per_day = st.slider('Cigarettes per day', min_value=0, max_value=100, value=20)
bpm = st.selectbox('On BP Meds (0 for No, 1 for Yes)', [0, 1])
prevalent_stroke = st.selectbox('Prevalent Stroke (0 for No, 1 for Yes)', [0, 1])
prevalent_hyp = st.selectbox('Prevalent Hypertension (0 for No, 1 for Yes)', [0, 1])
diabetes = st.selectbox('Diabetes (0 for No, 1 for Yes)', [0, 1])
tot_chol = st.slider('Total Cholesterol', min_value=100, max_value=400, value=200)
sys_bp = st.slider('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
dia_bp = st.slider('Diastolic Blood Pressure', min_value=40, max_value=150, value=80)
bmi = st.slider('BMI (Body Mass Index)', min_value=10.0, max_value=50.0, value=25.0)
heart_rate = st.slider('Heart Rate', min_value=40, max_value=120, value=70)
glucose = st.slider('Glucose', min_value=50, max_value=200, value=100)

# Prepare the feature set for prediction
features = np.array([male, age, current_smoker, cigs_per_day, bpm, prevalent_stroke, prevalent_hyp, 
                     diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose])

# Predict and display result
if st.button('Predict'):
    result = predict_heart_disease(features, threshold=0.17)
    if result == 1:
        st.write("The person is likely to have heart disease.")
    else:
        st.write("The person is unlikely to have heart disease.")
