<h1 align="center"><b>Kalkulori</b>: Sistem Rekomendasi Makanan</h1>
<div align="center">
  <img src="/Image/Logo_Kalkulori.png" width="300" alt="Logo Kalkulori" />
  <h4>Aplikasi web rekomendasi makanan menggunakan pendekatan berbasis konten dengan Scikit-Learn, FastAPI, dan TensorFlow.</h4>
</div>

---

## ℹ️ Info Umum

📌 **Kalkulori** adalah sistem rekomendasi makanan cerdas yang dirancang untuk membantu pengguna mencapai tujuan diet mereka dengan menyediakan rencana makan dan saran makanan yang dipersonalisasi. Di dunia yang makin peduli kesehatan saat ini, menjaga pola makan seimbang sangat penting. Proyek ini memanfaatkan *machine learning* untuk menawarkan rekomendasi yang tepat, mengatasi keterbatasan sistem rekomendasi diet yang canggih.

🔍 Sistem ini menggunakan **pendekatan berbasis konten**, yang berarti sistem menganalisis kandungan nutrisi, bahan, dan kata kunci resep untuk membuat rekomendasi. Pendekatan ini sangat efektif karena:
* 🎯 Tidak memerlukan data dari pengguna lain untuk memulai.
* 💡 Memberikan rekomendasi yang sangat relevan dengan pengguna individu.
* ❄️ Membantu menghindari masalah "mulai dingin" (*cold start*) yang sering ditemukan dalam sistem penyaringan kolaboratif.
* 📝 Menawarkan transparansi dalam rekomendasinya.

### ⚠️ Tantangan yang Dihadapi

Meskipun penyaringan berbasis konten memiliki banyak keuntungan, sistem ini juga datang dengan tantangan, seperti:
* **🔄 Kurangnya kebaruan dan keragaman**: Sistem mungkin merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, berpotensi membatasi paparan makanan baru.
* **⚙️ Skalabilitas**: Menangani sejumlah besar item dan fitur mereka bisa menjadi intensif secara komputasi.
* **📊 Kualitas atribut**: Rekomendasi sangat bergantung pada keakuratan dan konsistensi atribut makanan (kata kunci, info gizi).

---

## ✨ Fitur

Kalkulori menawarkan fitur-fitur utama berikut melalui *endpoint* API-nya:

### 🍽️ Suggestion Meal
* **🔍 Saran Berbasis Kata Kunci**: Merekomendasikan resep berdasarkan **kata kunci** yang diberikan pengguna (misalnya, 'tinggi protein', 'asia', 'vegetarian') dan **jumlah kalori target**.
* **📊 Pengelompokan Kalori (*Calorie Binning*)**: Memanfaatkan model TensorFlow yang telah dilatih sebelumnya untuk memprediksi kelompok kalori yang paling mungkin untuk resep, membantu dalam menemukan resep yang mendekati target kalori.
* **🎚 Peringkat & Randomisasi Cerdas**: Memberi peringkat saran berdasarkan skor kecocokan kata kunci dan probabilitas kalori yang diprediksi, kemudian mengambil sampel secara acak dari kumpulan kandidat teratas untuk memastikan keragaman.

### 📅 Meal Plan
* **🎯 Rencana Makan yang Dipersonalisasi**: Menghasilkan rencana makan sehari penuh (Sarapan, Makan Siang, Makan Malam, ditambah makanan tambahan opsional) berdasarkan **target total kalori** yang ditentukan.
* **⚖️ Toleransi Kalori**: Memungkinkan persentase toleransi yang dapat dikonfigurasi di sekitar target kalori untuk menemukan kombinasi makanan yang sesuai.
* **🏆 Pemilihan Berbasis Prioritas**: Resep diprioritaskan berdasarkan jenis makanan (Sarapan, Makan Siang, Makan Malam) dan kata kunci spesifik/umum untuk memastikan pilihan yang relevan.
* **📋Output Resep Terperinci**: Setiap makanan dalam rencana menyertakan detail resep yang komprehensif seperti bahan, waktu memasak, dan rincian nutrisi lengkap.

### 🔎 Search Meal
* **🔤 Pencarian Nama Resep**: Memungkinkan pengguna mencari resep berdasarkan **nama** sebagian atau penuh.
* **📝 Detail Resep Komprehensif**: Menyediakan detail lengkap untuk setiap resep melalui ID-nya, termasuk bahan, waktu memasak, dan semua informasi nutrisi.

---

## 🛠️ Pengembangan

### 🧹 Pemrosesan Data & Pelatihan Model

Sistem ini menggunakan **kumpulan data Kaggle Food.com**, yang berisi lebih dari 500.000 resep.
* **Pra-pemrosesan**: Resep dibersihkan, dan kata kunci di-binerkan menggunakan `MultiLabelBinarizer`. Fitur numerik seperti `Calories`, `ProteinContent`, dan `CarbohydrateContent` digunakan untuk pengelompokan dan prediksi.
* **Pengelompokan Kalori**: Kalori dikategorikan ke dalam kelompok spesifik (misalnya, '0-200', '201-400', dll.) dan dikodekan menggunakan `LabelEncoder`.
* **Prioritisasi Jenis Makanan**: Resep diberi `MealType` (Sarapan, Makan Siang, Makan Malam, Lain-lain) dan `MealPriority` berdasarkan kata kunci spesifik dan umum untuk memfasilitasi pembuatan rencana makan yang relevan.

### 🤖 Model Meal Plan (`MealPlanGenerator`)
* **Pengelompokan (*Clustering*)**: `MealPlanGenerator` menggunakan **pengelompokan KMeans** (dengan `StandardScaler` untuk fitur numerik) untuk mengelompokkan resep berdasarkan kandungan nutrisinya (Kalori, Protein, Karbohidrat). Ini membantu dalam memilih resep yang bervariasi dari kelompok yang berbeda untuk rencana makan.

### 🤖 Model Suggestion Meal (TensorFlow MLP)
* **Tugas Klasifikasi**: Model ini memprediksi kelompok kalori resep berdasarkan kandungan nutrisi dan fitur kata kuncinya.
* **Arsitektur**: **Multi-Layer Perceptron (MLP)** Sekuensial digunakan dengan aktivasi `relu` untuk lapisan tersembunyi dan `softmax` untuk lapisan keluaran guna memprediksi probabilitas di seluruh kelompok kalori.
* **Pelatihan**: Dilatih menggunakan *loss* `SparseCategoricalCrossentropy` dan *optimizer* `Adam`, dengan bobot kelas untuk menangani potensi ketidakseimbangan kelas.

### 🚀 Deployment dengan AWS

1. **Persiapan Lingkungan Produksi**: Aplikasi disiapkan untuk lingkungan produksi menggunakan Gunicorn. Gunicorn bertindak sebagai server WSGI yang tangguh, membantu mengelola worker aplikasi FastAPI. Ini memungkinkan aplikasi untuk menangani berbagai permintaan secara bersamaan dengan lebih efisien dan stabil.

2. **Manajemen Layanan Otomatis**: Untuk memastikan aplikasi selalu aktif dan secara otomatis berjalan kembali setelah server dihidupkan ulang, digunakan systemd. Ini adalah sistem init dan system manager pada Linux yang memungkinkan konfigurasi aplikasi sebagai layanan di background. Dengan demikian, aplikasi dapat dikelola dan dijalankan secara otomatis.

3. **Akses dan Pengujian**: Setelah berhasil di-deploy, aplikasi dapat diakses melalui alamat IP publik instans EC2 Anda. Untuk menguji fungsionalitasnya, endpoint API (seperti rekomendasi resep dan detail makanan) dapat diakses menggunakan tool seperti curl atau langsung melalui peramban web.

4. **Pembaruan dan Pemeliharaan**: Setiap kali ada pembaruan pada konfigurasi atau kode aplikasi, layanan dapat dengan mudah di-reload atau di-restart menggunakan perintah systemctl. Hal ini memungkinkan penerapan perubahan secara cepat tanpa mengganggu ketersediaan layanan dalam waktu lama.

---

## 💻 Teknologi

Proyek ini dikembangkan menggunakan teknologi-teknologi berikut:

* **Python**: 3.10.8
* **FastAPI**: 0.115.12 (untuk membangun API web)
* **Uvicorn**: 0.34.3 (server ASGI untuk FastAPI)
* **Scikit-learn**: 1.3.2 (untuk MultiLabelBinarizer, LabelEncoder, StandardScaler, KMeans)
* **TensorFlow**: 2.13.0 (untuk model prediksi kelompok kalori)
* **Pandas**: 2.2.3 (untuk manipulasi data)
* **NumPy**: 1.26.0 (untuk operasi numerik)
* **XGBoost**: 3.0.2 (model gradient boosting yang mungkin digunakan)
* **Pydantic**: 2.11.5 (untuk validasi data dan manajemen setting di FastAPI)
* **Seaborn**: 0.12.2 (untuk visualisasi data statistik)
* **Matplotlib**: 3.10.3 (untuk pembuatan plot dan visualisasi data dasar)
* **Gunicorn**: 23.0.0 (server WSGI untuk deployment FastAPI)
* **Joblib**: (untuk menyimpan/memuat model KMeans dan StandardScaler)
* **Pickle**: (untuk menyimpan/memuat MultiLabelBinarizer, LabelEncoder, bins_labels, model_input_feature_columns)

<br/>

<img src = "https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white"> <img src = "https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"> <img src = "https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white"> <img src = "https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img src = "https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"> <img src = "https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">

---

## ⚡ Setup

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

### Kloning Repositori
```bash
git clone https://github.com/Akmal7604/Capstone_Akmal_Andri_Zahran.git
```

### Masuk ke file project
```bash
cd Capstone_Akmal_Andri_Zahran
```

### Buat dan Aktifkan Virtual Environment
```bash
cd Meal_Plan_Search
cd Meal_Plan/ # Atau Search/ atau Suggestion_Meal/
python3 -m venv venv
source venv/bin/activate # Di Windows: .\venv\Scripts\activate
```

### Instal Dependensi
```bash
pip install -r requirements.txt
```

### Jalankan Aplikasi FastAPI
```bash
fastapi dev main.py # cd Meal_Plan, bisa menggunakan search ataupun suggestion_meal
```
