# Stress Prediction using Deep Learning (LSTM) & NCF

Project ini berisi notebook Jupyter untuk melakukan **prediksi / klasifikasi tingkat stres karyawan** menggunakan:

- **Model Deep Learning berbasis LSTM (Long Short-Term Memory)**
- **Model NCF (Neural Collaborative Filtering)** yang memanfaatkan `employee_id` dan `department`.

Setiap model dievaluasi menggunakan:
- **Accuracy**
- **Classification Report** (precision, recall, f1-score)
- **Confusion Matrix** (dalam bentuk angka dan visual heatmap).

---

## 1. Struktur Proyek

File utama yang digunakan:

- `stress_prediction_LSTM_NCF_clean.ipynb`  
  Notebook Jupyter yang berisi:
  - Load dan eksplorasi dataset
  - Preprocessing fitur untuk model LSTM dan NCF
  - Pembangunan arsitektur model LSTM
  - Pembangunan arsitektur model NCF
  - Training masing-masing model
  - Evaluasi model (akurasi, classification report, confusion matrix)

- `dataset_prediksi_stres_1000_balanced.csv`  
  Dataset yang digunakan dalam notebook.

> **Catatan:** Pastikan file dataset berada di direktori yang sama dengan notebook saat dijalankan.

---

## 2. Dataset

Notebook menggunakan file:

```text
dataset_prediksi_stres_1000_balanced.csv
```

Secara umum, dataset memiliki kolom:

- `employee_id` – ID unik karyawan (kategori)
- `department` – departemen tempat karyawan bekerja (kategori)
- `workload` – beban kerja
- `work_life_balance` – skor keseimbangan kerja–hidup
- `team_conflict` – tingkat konflik dalam tim
- `management_support` – dukungan manajemen
- `work_environment` – kualitas lingkungan kerja
- `stress_level` – skor tingkat stres (kontinu)
- `label` – kelas stres (misalnya 0, 1, 2) yang digunakan sebagai **target klasifikasi**

Kolom **`label`** adalah target (y) yang diprediksi oleh kedua model (LSTM dan NCF).

---

## 3. Lingkungan & Dependensi

Project ini menggunakan Python 3.x dan beberapa library berikut:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` (yang sudah termasuk `tf.keras`)

### Cara Install Dependensi

Gunakan perintah berikut (sesuaikan dengan environment Anda):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Disarankan menggunakan virtual environment (conda/venv) agar dependensi lebih terkontrol.

---

## 4. Alur Notebook

Secara garis besar, alur di dalam notebook adalah sebagai berikut:

### 4.1. Import Library

Semua library yang dibutuhkan akan di-import di awal:

- Library untuk data: `numpy`, `pandas`
- Visualisasi: `matplotlib`, `seaborn`
- Preprocessing & evaluasi: `sklearn`
- Deep Learning: `tensorflow.keras`

---

### 4.2. Load & Inspect Dataset

- Membaca dataset:

  ```python
  df = pd.read_csv('dataset_prediksi_stres_1000_balanced.csv')
  ```

- Menampilkan:
  - Ukuran data (`df.shape`)
  - Contoh beberapa baris awal (`df.head()`)
  - Informasi tipe data (`df.info()`)

Tujuan: memastikan data sudah terbaca dengan benar dan struktur kolom sesuai.

---

### 4.3. Preprocessing Data

#### Fitur untuk Model LSTM

Fitur numerik yang digunakan:

```python
feature_cols = [
    'workload',
    'work_life_balance',
    'team_conflict',
    'management_support',
    'work_environment',
    'stress_level'
]
target_col = 'label'
```

Langkah preprocessing:

1. Ambil fitur (`X_num`) dan target (`y`).
2. Lakukan `train_test_split` untuk membagi data menjadi train dan test.
3. Lakukan **standardisasi** fitur numerik menggunakan `StandardScaler`.
4. Bentuk ulang data menjadi format yang sesuai LSTM:
   - Dari `(n_samples, n_features)` menjadi `(n_samples, n_timesteps, n_features_per_timestep)`
   - Dalam notebook ini: 6 fitur dijadikan 6 timesteps, masing-masing dengan 1 fitur → `(n_samples, 6, 1)`.
5. Lakukan **one-hot encoding** pada label dengan `to_categorical`.

#### Fitur untuk Model NCF

Untuk NCF digunakan fitur kategorikal:

- `employee_id` → user
- `department` → item

Langkah:

1. Encode `employee_id` menggunakan `LabelEncoder` menjadi indeks user.
2. Encode `department` menjadi indeks item.
3. Hitung banyaknya user unik (`n_users`) dan item unik (`n_items`) untuk ukuran embedding.
4. Lakukan `train_test_split` yang konsisten dengan data numerik dan label.

---

## 5. Model Deep Learning (LSTM)

### 5.1. Arsitektur Model

Secara garis besar:

- Input shape: `(6, 1)`
- Layer utama:
  - `LSTM(64, return_sequences=True)`
  - `LSTM(32)`
  - `Dense(32, activation='relu')`
  - `Dense(num_classes, activation='softmax')` sebagai output

Model dikompilasi dengan:

```python
loss = 'categorical_crossentropy'
optimizer = Adam(learning_rate=0.001)
metrics = ['accuracy']
```

### 5.2. Training

Model dilatih dengan:

- `epochs`: sekitar 30
- `batch_size`: sekitar 32
- `validation_split`: 0.1

### 5.3. Evaluasi

Setelah training:

1. Model melakukan prediksi probabilitas pada data test.
2. Probabilitas dikonversi menjadi kelas dengan `argmax`.
3. Digunakan fungsi `evaluate_classification(...)` untuk menampilkan:
   - Accuracy
   - Classification report
   - Confusion Matrix (angka)
   - Confusion Matrix heatmap dengan `seaborn.heatmap`

---

## 6. Model NCF (Neural Collaborative Filtering)

### 6.1. Arsitektur Model

Input:

- `user_input` (1 dimensi, berisi indeks user)
- `item_input` (1 dimensi, berisi indeks department)

Langkah:

1. Masing-masing input masuk ke layer `Embedding`:
   - `Embedding(n_users, user_embedding_dim)`
   - `Embedding(n_items, item_embedding_dim)`
2. Hasil embedding di-`Flatten`.
3. Vektor user dan item di-`Concatenate`.
4. Melewati beberapa `Dense` layer (misalnya 64 dan 32 neuron).
5. Output akhir: `Dense(num_classes, activation='softmax')`.

Model juga dikompilasi dengan:

```python
loss = 'categorical_crossentropy'
optimizer = Adam(learning_rate=0.001)
metrics = ['accuracy']
```

### 6.2. Training

Model dilatih dengan input:

```python
[user_train, item_train], y_train_cat
```

dengan parameter serupa (epochs, batch_size, validation_split).

### 6.3. Evaluasi

Proses evaluasi sama seperti LSTM:

1. Prediksi probabilitas → `argmax` menjadi kelas.
2. Hitung:
   - Accuracy
   - Classification report
   - Confusion matrix + heatmap.

---

## 7. Evaluasi & Perbandingan Hasil

Dari notebook, kamu bisa:

- Membandingkan akurasi antara:
  - Model LSTM
  - Model NCF
- Melihat kekuatan/kelemahan masing-masing model pada kelas tertentu melalui **confusion matrix** dan **classification report**.

Beberapa hal yang bisa diamati:

- Apakah model lebih baik mengenali kelas stres tertentu (misal kelas tinggi atau rendah)?
- Apakah NCF (berbasis kombinasi `employee_id` dan `department`) memberi hasil berbeda dibanding LSTM yang berbasis fitur numerik?

---

## 8. Cara Menjalankan Notebook

1. Pastikan semua dependensi sudah ter-install.
2. Pastikan file berikut berada di direktori yang sama:
   - `stress_prediction_LSTM_NCF_clean.ipynb`
   - `dataset_prediksi_stres_1000_balanced.csv`
3. Buka terminal dan jalankan Jupyter:

   ```bash
   jupyter notebook
   ```

4. Buka file `stress_prediction_LSTM_NCF_clean.ipynb`.
5. Jalankan cell dari atas ke bawah secara berurutan (Restart & Run All juga bisa).

---

## 9. Pengembangan Lanjutan (Optional)

Beberapa ide pengembangan:

- Menambahkan model lain (misal CNN 1D untuk fitur numerik).
- Hyperparameter tuning (learning rate, jumlah neuron, jumlah epoch).
- Cross-validation untuk validasi yang lebih kuat.
- Menyimpan model terlatih (`model.save(...)`) dan membuat API sederhana (misalnya dengan Flask/FastAPI) untuk prediksi real-time.

---

## 10. Lisensi

Silakan sesuaikan bagian ini sesuai kebutuhan (misal: pribadi, akademik, atau open source).

```text
Project ini dibuat untuk keperluan pembelajaran dan/atau tugas akademik
terkait penerapan Deep Learning dan NCF pada prediksi tingkat stres karyawan.
```
