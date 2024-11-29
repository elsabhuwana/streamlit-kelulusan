import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup Layout Responsif
st.set_page_config(
    page_title="Prediksi Kelulusan Siswa",
    layout="wide"
)

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r'D:\KULIAH\SEMESTER5\Data Sience\Tugas1\venv\Minggu13\student_data.csv') 
    return data

data = load_data()

# Navbar
st.sidebar.title("PILIH MENU")
selection = st.sidebar.radio("", ["Deskripsi", "Dataset", "Grafik", "Prediksi", "Bantuan"])

# 1. Deskripsi Aplikasi
if selection == "Deskripsi":
    st.title("Prediksi Kelulusan Siswa")
    st.write("""
    Aplikasi Web Prediksi Kelulusan Siswa adalah sebuah aplikasi berbasis web yang dikembangkan menggunakan framework Streamlit. Aplikasi ini dirancang untuk menganalisis data siswa dan memprediksi kelulusan mereka berdasarkan beberapa parameter seperti usia, waktu belajar, jumlah kegagalan sebelumnya, absensi, serta nilai pada G1 dan G2 nilai sebelumnya, Dataset yang digunakan dalam aplikasi ini memuat informasi lengkap tentang siswa, termasuk data akademik, kebiasaan belajar, dan faktor lain yang berkontribusi terhadap performa mereka. Dalam aplikasi ini, data dianalisis dan divisualisasikan melalui grafik distribusi nilai, hubungan antar fitur, dan heatmap korelasi untuk memudahkan pengguna memahami pola yang ada dalam data. Aplikasi ini menggunakan Metode Random Forest Classifier sebagai model prediksi. Random Forest adalah algoritma pembelajaran mesin berbasis ensemble yang kuat dan mampu memberikan hasil prediksi yang andal dengan tingkat akurasi yang tinggi. Model ini memprediksi kelulusan berdasarkan nilai akhir siswa (G3) dengan membagi siswa menjadi dua kategori Lulus dan Tidak Lulus
    """)
    
# 2. Melihat Dataset
elif selection == "Dataset":
    st.title("Dataset Siswa")
    st.write("""
    Halaman ini menampilkan dataset yang digunakan dalam aplikasi ini. Dataset ini berisi berbagai fitur yang digunakan 
    untuk memprediksi kelulusan siswa, seperti usia, waktu belajar, jumlah kegagalan sebelumnya, absensi, dan nilai-nilai G1 dan G2.
    Silakan pilih kolom yang ingin ditampilkan di bawah ini.
    """)
    selected_columns = st.multiselect("Pilih kolom yang ingin ditampilkan:", data.columns, default=data.columns)
    st.dataframe(data[selected_columns])

# 3. Grafik
elif selection == "Grafik":
    st.title("Grafik Data Siswa")
    st.write("""
    Di halaman ini, Anda dapat melihat berbagai visualisasi grafik yang membantu dalam memahami pola dalam dataset.
    Grafik-grafik ini meliputi distribusi nilai siswa, hubungan antara absensi dan kelulusan, serta korelasi antar fitur.
    """)

    # Grafik Distribusi Nilai
    st.subheader("Distribusi Nilai (G1, G2, G3)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['G1'], kde=True, ax=ax[0])
    sns.histplot(data['G2'], kde=True, ax=ax[1])
    sns.histplot(data['G3'], kde=True, ax=ax[2])
    ax[0].set_title('Distribusi G1')
    ax[1].set_title('Distribusi G2')
    ax[2].set_title('Distribusi G3')
    st.pyplot(fig)

    # Grafik Absensi dan G3
    st.subheader("Hubungan Absensi dengan Nilai G3")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=data['absences'], y=data['G3'], ax=ax)
    ax.set_title("Absensi vs Nilai Akhir (G3)")
    st.pyplot(fig)

    # Grafik Studytime dan G3
    st.subheader("Hubungan Studytime dengan Nilai G3")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=data['studytime'], y=data['G3'], ax=ax)
    ax.set_title("Studytime vs Nilai Akhir (G3)")
    st.pyplot(fig)

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 4. Prediksi
elif selection == "Prediksi":
    st.title("Prediksi Kelulusan Siswa")
    st.write("""
    Pada halaman ini, Anda dapat memprediksi kelulusan siswa berdasarkan data input yang Anda berikan.
    Masukkan data siswa seperti usia, waktu belajar, jumlah kegagalan sebelumnya, absensi, serta nilai G1 dan G2 untuk 
    mendapatkan prediksi apakah siswa tersebut lulus atau tidak.
    """)

    # Fitur untuk Prediksi
    features = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']
    X = data[features]
    y = (data['G3'] > 10).astype(int)  # Kelulusan: G3 > 10 = 1 (Lulus), <= 10 = 0 (Tidak Lulus)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Menampilkan Akurasi dan Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi Model: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, target_names=["Tidak Lulus", "Lulus"])
    st.text("Laporan Evaluasi Model:")
    st.text(report)

    # Input Pengguna
    st.subheader("Masukkan Data Siswa untuk Prediksi")
    age = st.slider("Age", 10, 20, 15)
    studytime = st.slider("Studytime (1-4)", 1, 4, 2)
    failures = st.slider("Failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 100, 10)
    G1 = st.slider("G1", 0, 20, 10)
    G2 = st.slider("G2", 0, 20, 10)

    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2]
    })

    # Prediksi
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success("Siswa diprediksi LULUS")
    else:
        st.error("Siswa diprediksi TIDAK LULUS")


# 5. Bantuan
elif selection == "Bantuan":
    st.title("Bantuan Penggunaan Aplikasi")
    st.write("1. Gunakan navigasi di sidebar untuk berpindah antar halaman.")
    st.write("2. Halaman Deskripsi memberikan informasi tentang tujuan dan cara kerja aplikasi.")
    st.write("3. Halaman Dataset menampilkan data siswa yang digunakan.")
    st.write("4. Halaman Grafik menyediakan visualisasi data penting.")
    st.write("5. Halaman Prediksi memungkinkan Anda memprediksi kelulusan siswa berdasarkan input data.")
