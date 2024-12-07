# Import library
import streamlit as st
import pandas as pd
import pickle

# Judul aplikasi
st.title("Prediksi Penggunaan Sepeda Berdasarkan Waktu dan Cuaca")

# Load model dengan benar
model = pickle.load("model/model.pkl")

# Sidebar untuk input prediksi
st.sidebar.header("Input Data untuk Prediksi")
temp = st.sidebar.slider("Suhu (0-100)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Kelembaban (0-1)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Kecepatan Angin (0-1)", 0.0, 1.0, 0.5)
season = st.sidebar.selectbox("Musim", ["Winter", "Spring", "Summer", "Fall"])
season_mapping = {"Winter": [0, 0, 0], "Spring": [1, 0, 0], "Summer": [0, 1, 0], "Fall": [0, 0, 1]}
season_features = season_mapping[season]

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Gabungkan input fitur
    input_features = [[temp, hum, windspeed] + season_features + [0, 0, 0]]  # weekday dummy input
    prediction = model.predict(input_features)[0]

    # Tampilkan hasil prediksi
    st.sidebar.write(f"**Hasil Prediksi Jumlah Pengguna Sepeda: {int(prediction)}**")
