import streamlit as st
import pickle
import numpy as np

pred_model = pickle.load(open("prediction_model.pkl", "rb"))

st.title("Job Placement Prediction Model")

input_txt = st.text_input("Enter details...")

if input_txt:
    separateInputByComma = input_txt.split(",") 
    df = np.asarray(separateInputByComma, dtype=float)
    prediction = pred_model.predict(df.reshape(1, -1))

    if prediction[0] == 1:
        st.write("The chances of the student getting placed is high.")
    
    else: 
        st.write("The chances of the student getting placed is low.")