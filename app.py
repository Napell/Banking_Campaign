#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle


# Load the model and scaler
try:
    model = pickle.load(open('log_model.pkl', 'rb'))
    st.success("Model loaded successfully")
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    with open("columns.pkl","rb") as f:
        features = pickle.load(f)
    st.success("column loaded successfully")
except FileNotFoundError:
    st.error("File not found")
except Exception as e:
    st.error("could not load features")

st.title("Bank Deposit Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
job = st.selectbox("Job", ['admin.',
'blue-collar','technician','services','management','retired','entrepreneur','self-employed','housemaid','unemployed','student','unknown'])
marital=st.selectbox("Marital", ['single','married','divorce','unknown'])
education = st.selectbox("Education", ['university.degree','high.school','basic.9y','professional.course','basic.4y','basic.6y','unknown','illiterate'])
default = st.selectbox("Default", ['yes','no','unknown'])
housing = st.selectbox("Housing", ['yes','no','unknown'])
loan = st.selectbox("loan", ['yes','no','unknown'])
contact = st.selectbox("contact", ['cellular','telephone'])
month = st.selectbox("month", ['may','jul','aug','jun','nov','apr','oct','sep','mar','dec'])
day_of_week = st.selectbox("day_of_week", ['thu','mon','wed','tue','fri'])
duration = st.number_input("duration", min_value=1, max_value=3800)
campaign= st.number_input("campaign", min_value=1, max_value=40)
pdays = st.number_input("pdays", min_value=1, max_value=999)
previous = st.number_input("previous", min_value=1, max_value=7)
poutcome = st.selectbox("poutcome", ['nonexistent','failure',
'success'])
emp_var_rate = st.number_input("emp_var_rate")
cons_price_idx= st.number_input("cons_price_idx")
cons_conf_idx = st.number_input("cons_conf_idx")
euribor3m = st.number_input("euribor3m")
nr_employed = st.number_input("nr_employed")

            

if st.button("Predict"):
    input_df = pd.DataFrame([{
        'age': age,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx':cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'job': job,
        'marital':marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact':contact,
        'month':month,
        'day_of_week': day_of_week,
        'poutcome': poutcome,
      
      
      
        
    }])

    input_df = pd.get_dummies(input_df)
  
   
    for col in features:

        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    st.write(input_df)


    input_df_scaled = scaler.transform(input_df)
    # st.write(input_df_scaled)
    st.write(model.predict(input_df_scaled))
    prediction = model.predict(input_df_scaled)[0]
    st.write(model.coef_)
    
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")



    

