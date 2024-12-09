# Import library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Judul aplikasi
st.title("Prediksi Penggunaan Sepeda Berdasarkan Waktu dan Cuaca")

# Preprocessing dan Pelatihan Model
@st.cache_resource  # Cache agar model tidak dilatih ulang setiap kali aplikasi di-refresh
def train_model():
    # Baca dataset
    data = pd.DataFrame({
        "temp": np.random.uniform(0.2, 1.0, 100),
        "hum": np.random.uniform(0.3, 0.8, 100),
        "windspeed": np.random.uniform(0.1, 0.6, 100),
        "season": np.random.choice([2, 3, 4], 100),  # Dummy data
        "weekday": np.random.choice([1, 2, 3], 100),
        "cnt": np.random.randint(50, 500, 100),
    })
    
    # Preprocessing data
    data = pd.get_dummies(data, columns=["season", "weekday"], drop_first=True)
    scaler = MinMaxScaler()
    data[["temp", "hum", "windspeed"]] = scaler.fit_transform(data[["temp", "hum", "windspeed"]])

    # Fitur dan target
    X = data[["temp", "hum", "windspeed", "season_3", "season_4", "weekday_2", "weekday_3"]]
    y = data["cnt"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Memuat model
model = train_model()

# Sidebar untuk input prediksi
st.sidebar.header("Input Data untuk Prediksi")
temp = st.sidebar.slider("Suhu (0-100)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Kelembaban (0-1)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Kecepatan Angin (0-1)", 0.0, 1.0, 0.5)
season = st.sidebar.selectbox("Musim", ["Winter", "Spring", "Summer", "Fall"])
season_mapping = {"Winter": [0, 0], "Spring": [1, 0], "Summer": [0, 1], "Fall": [0, 0]}
season_features = season_mapping[season]

weekday = st.sidebar.selectbox("Hari Kerja", ["Monday", "Tuesday", "Wednesday"])
weekday_mapping = {"Monday": [0, 0], "Tuesday": [1, 0], "Wednesday": [0, 1]}
weekday_features = weekday_mapping[weekday]

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Gabungkan input fitur
    input_features = [[temp, hum, windspeed] + season_features + weekday_features]
    prediction = model.predict(input_features)[0]

    # Tampilkan hasil prediksi
    st.sidebar.write(f"**Hasil Prediksi Jumlah Pengguna Sepeda: {int(prediction)}**")