import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\rohit\AVS CODE\CLASSIFICATION\Logistic_regression.pkl', 'rb'))

# Set the title of the app
st.title("Vehicle Prediction App")

# Add a brief description
st.write("""This app predicts whether a user will purchase a product based on their **age** and **estimated salary**.""")

# Input details
age = st.number_input("Age", min_value=0, max_value=100, step=1, value=25)
estimated_salary = st.number_input("Estimated Salary", min_value=0, step=2000, value=50000)

# Predict purchase function
def predict_purchase(age, estimated_salary):
    # Prepare the input data for prediction
    input_data = np.array([[age, estimated_salary]])
    prediction = model.predict(input_data)
    return prediction[0]

# Predict button and results display
if st.button("Predict"):
    prediction = predict_purchase(age, estimated_salary)
    
    # Display results
    if prediction == 0:
        st.error("The customer is not likely to purchase a vehicle.")
    else:
        st.success("The customer is likely to purchase a vehicle.")
