import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
pipe = load('pipe.pkl')

st.title("💻 Laptop Price Predictor")

# ✅ Hardcoded options
company_options = ['Dell', 'HP', 'Lenovo', 'Asus', 'Apple', 'MSI', 'Acer']
type_options = ['Notebook', 'Gaming', '2 in 1 Convertible', 'Ultrabook']
cpu_options = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD Ryzen 5', 'AMD Ryzen 7', 'Other']
gpu_options = ['Intel', 'Nvidia', 'AMD']
os_options = ['Windows', 'Mac', 'Linux', 'Other']

# Input fields (simplified)
company = st.selectbox('Company', company_options)
laptop_type = st.selectbox('Type', type_options)
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (kg)', format="%.2f")
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024, 2048])
cpu = st.selectbox('CPU Brand', cpu_options)
gpu = st.selectbox('GPU Brand', gpu_options)
os = st.selectbox('Operating System', os_options)

# Predict button
if st.button('Predict Price'):
    touchscreen_bin = 1 if touchscreen == 'Yes' else 0
    ips_bin = 1 if ips == 'Yes' else 0

    # Dummy values for missing features
    product_name = "Unknown"
    ppi = 0.0

    # ✅ Match model's expected columns
    input_data = pd.DataFrame([[company, product_name, laptop_type, ram, weight, touchscreen_bin, ips_bin,
                                ppi, cpu, hdd, ssd, gpu, os]],
                              columns=['Company', 'Product', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
                                       'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    predicted_price = np.exp(pipe.predict(input_data)[0])
    st.success(f"💰 Estimated Price: ₹{round(predicted_price, 2)}")



