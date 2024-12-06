# Import library
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Judul aplikasi
st.title("Prediksi Penggunaan Sepeda Berdasarkan Waktu dan Cuaca")

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset yang Diunggah")
    st.write(data.head())

    # Informasi dataset
    st.markdown("### Informasi Dataset")
    st.write(data.info())
    st.write(data.describe())

    # Data Wrangling
    st.markdown("### Data Wrangling")
    st.write("Melakukan one-hot encoding dan normalisasi data...")
    data = pd.get_dummies(data, columns=['season', 'weekday', 'weathersit'], drop_first=True)
    scaler = MinMaxScaler()
    data[['temp', 'hum', 'windspeed']] = scaler.fit_transform(data[['temp', 'hum', 'windspeed']])

    # Menambahkan kategori musim
    season_mapping = {'season_2': 'Spring', 'season_3': 'Summer', 'season_4': 'Fall'}
    data['season_mapped'] = data[[col for col in data.columns if col.startswith('season_')]].idxmax(axis=1)
    data['season_mapped'] = data['season_mapped'].map(lambda x: season_mapping.get(x, 'Winter'))

    st.write("Data setelah preprocessing:")
    st.write(data.head())

    # Visualisasi EDA
    st.markdown("### Visualisasi EDA")

    # Distribusi pengguna sepeda berdasarkan musim
    st.subheader("Distribusi Penggunaan Sepeda Berdasarkan Musim")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='season_mapped', y='cnt', data=data, ax=ax1)
    plt.title('Distribusi Penggunaan Sepeda Berdasarkan Musim')
    st.pyplot(fig1)

    # Scatter plot suhu vs jumlah pengguna
    st.subheader("Hubungan Suhu dengan Jumlah Pengguna")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='temp', y='cnt', data=data, ax=ax2)
    plt.title('Hubungan Suhu dengan Penggunaan Sepeda')
    st.pyplot(fig2)

    # Heatmap korelasi antar variabel
    st.subheader("Heatmap Korelasi Antar Variabel")
    corr = data[['temp', 'hum', 'windspeed', 'cnt']].corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Pemodelan
    st.markdown("### Pemodelan Prediksi")

    # Fitur dan target
    X = data[['temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4', 'weekday_1', 'weekday_2', 'weekday_3']]
    y = data['cnt']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("**Evaluasi Model:**")
    st.write(f"- Mean Squared Error (MSE): {mse}")
    st.write(f"- R-squared (R2): {r2}")

    # Prediksi dengan input
    st.sidebar.subheader("Input Data untuk Prediksi")
    temp = st.sidebar.slider("Suhu (0-1)", 0.0, 1.0, 0.5)
    hum = st.sidebar.slider("Kelembaban (0-1)", 0.0, 1.0, 0.5)
    windspeed = st.sidebar.slider("Kecepatan Angin (0-1)", 0.0, 1.0, 0.5)
    season = st.sidebar.selectbox("Musim", ["Winter", "Spring", "Summer", "Fall"])
    season_mapping_input = {"Winter": [0, 0, 0], "Spring": [1, 0, 0], "Summer": [0, 1, 0], "Fall": [0, 0, 1]}
    season_features = season_mapping_input[season]

    # Prediksi berdasarkan input
    if st.sidebar.button("Prediksi"):
        input_features = [[temp, hum, windspeed] + season_features + [0, 0, 0]]  # weekday dummy input
        prediction = model.predict(input_features)[0]
        st.sidebar.write(f"**Hasil Prediksi Jumlah Pengguna Sepeda: {int(prediction)}**")