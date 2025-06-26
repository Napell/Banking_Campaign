#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle


# Load the model and scaler
model = pickle.load(open('log_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

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

features=['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_married', 'marital_single', 'marital_unknown', 'education_basic.6y', 'education_basic.9y', 'education_high.school', 'education_illiterate', 'education_professional.course', 'education_university.degree', 'education_unknown', 'default_unknown', 'default_yes', 'housing_unknown', 'housing_yes', 'loan_unknown', 'loan_yes', 'contact_telephone', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed', 'poutcome_nonexistent', 'poutcome_success'
              ]

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
    prediction = model.predict(input_df_scaled)[0]
    st.write(model.predict(input_df_scaled))
    st.write(prediction)
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")



    

