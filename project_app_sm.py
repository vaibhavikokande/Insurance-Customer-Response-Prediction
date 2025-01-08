import streamlit as st  # for building interactive web applications 
import pandas as pd     # Used for handling data structure
from joblib import load  # to load the pre-trained model saved earlier

# Load the trained AdaBoost model
try:
    model = load('insurance_model_sm.joblib')  # Ensure the file path is correct
except FileNotFoundError:
    st.error("Model file 'insurance_model.joblib' not found. Please ensure the file is in the correct location.")
    st.stop()
# If the file is missing, an error message is displayed, and the app stops further execution (st.stop()).


# Create a Streamlit app
st.title("Insurance Customer Response Prediction")
st.write("Use this app to predict whether a customer will respond to an insurance offer.")
# Define the title and introductory text for the app.


#create Input fields for feature values
st.header("Customer Information")

# Input for Vehicle Damage
Vehicle_Damage = st.selectbox("Vehicle Damage or Not", ('YES', 'NO'))

# Input for Age
Age = st.number_input("Age of the Customer", min_value=0, max_value=120, value=25, step=1) # st.selectbox and st.number_input functions collect inputs from users


# Input for Vehicle Age
Vehicle_Age = st.selectbox("Vehicle Age", ('< 1 Year', '1-2 Year', '>2 Year'))

# Input for Driving License
Driving_License = st.selectbox("Driving License Status", ('Having', 'NOT'))

# Input for Gender
Gender = st.selectbox("Gender", ('Male', 'Female'))

# Input for Region Code
Region_Code = st.number_input("Region Code", min_value=0.0, max_value=100.0, value=1.0, step=1.0)

# Input for Annual Premium
Annual_Premium = st.number_input("Annual Premium Amount", min_value=0.0, max_value=1000000.0, value=20000.0, step=100.0)

Policy_Sales_Channel = st.number_input("Policy Channel")

Vintage=st.number_input("Vintage")

Previously_Insured=st.selectbox("Previously Insured",('Insured','Not Insured'))

# Categorical Inputs:
# - Vehicle_Damage: Whether the vehicle is damaged or not.
# - Vehicle_Age: Age of the vehicle (<1 Year, 1-2 Year, >2 Year).
# - Driving_License: If the customer has a valid driving license.
# - Gender: Male or Female.
# - Previously_Insured: If the customer is already insured.

# Numerical Inputs:
# - Age: Age of the customer.
# - Region_Code: Customer's region code.
# - Annual_Premium: Premium amount.
# - Policy_Sales_Channel: The channel used for selling the policy.
# - Vintage: Time the customer has been associated with the company.




# Map categorical inputs to numerical values
label_mapping = {
    'YES': 1,
    'NO': 0,
    'Having': 1,
    'NOT': 0,
    'Male': 1,
    'Female': 0,
    '< 1 Year': 0,
    '1-2 Year': 1,
    '>2 Year': 2,
    'Insured':1,
    'Not Insured':0
}

# Convert categorical inputs to numerical
Vehicle_Damage = label_mapping[Vehicle_Damage]
Vehicle_Age = label_mapping[Vehicle_Age]
Driving_License = label_mapping[Driving_License]
Gender = label_mapping[Gender]
Previously_Insured=label_mapping[Previously_Insured]

# Arrange features in the correct order
features = [[
    Driving_License, Vehicle_Damage, Vehicle_Age, Gender,
    Annual_Premium, Region_Code, Age,Annual_Premium,Vintage,Previously_Insured
]]

# Make a prediction using the model
try:
    prediction = model.predict(features) # Uses the model.predict() function to generate a prediction (0 for No Response, 1 for Response).

    # Display the prediction result
    st.header("Prediction Result")
    if prediction[0] == 0:
        st.success("The model predicts: No RESPONSE.")
    else:
        st.error("The model predicts: RESPONSE.")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
# If any error occurs during the prediction, it’s displayed on the app for debugging
