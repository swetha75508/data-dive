#!/usr/bin/env python
# coding: utf-8

# ## Deployment

# In[1]:

import numpy as np
import streamlit as st
import pickle
import pandas as pd


# In[2]:


# Load the trained model
file = 'Bankruptcy_Prevention.pkl'
with open(file, 'rb') as f:
    model = pickle.load(f)


# In[3]:


# Title
st.title("Bankruptcy Prevention Prediction")

# User input fields
st.sidebar.header("Company Financial Details")

# Load dataset to get column names
data = pd.read_excel('bankruptcy-prevention.xlsx', sheet_name=0, header=None)
data = data[0].str.split(';', expand=True)
data.columns = data.iloc[0].str.strip().str.lower()
data = data[1:].reset_index(drop=True)
feature_columns = data.drop(columns=['class']).columns

# User input dropdowns
user_input = {}
for col in feature_columns:
    unique_values = sorted(data[col].unique())
    user_input[col] = st.sidebar.selectbox(col, unique_values)

# Convert input into DataFrame
df = pd.DataFrame([user_input])

# Predict
if st.sidebar.button("Predict"):
    pred_prob = model.predict_proba(df)

    # Display bankruptcy probability
    st.subheader("Bankruptcy Probability")
    st.write("Yes" if pred_prob[0][1] >= 0.5 else "No")

    # Show predicted probabilities
    st.subheader("Predicted Probability")
    st.write(pred_prob)

