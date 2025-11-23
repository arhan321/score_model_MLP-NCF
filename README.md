
# Stress Level Prediction using MLP & NCF

This project uses machine learning and deep learning to classify employee stress levels
into three categories (Low, Medium, High). Two models are built and evaluated:

1. **MLP (Multilayer Perceptron)** – classification based on employee features  
2. **NCF (Neural Collaborative Filtering)** – embedding-based model using employee_id and department

---

## 📂 Dataset

File: `dataset_prediksi_stres_200.csv`

Columns include:
- employee_id
- department
- workload
- work_life_balance
- team_conflict
- management_support
- work_environment
- stress_level (target)

---

## 🎯 Objectives

- Convert numeric stress_level into 3 classes:
  - 0–20 → Low  
  - 21–40 → Medium  
  - 41–60 → High  

- Train MLP & NCF models  
- Evaluate accuracy and confusion matrices  

---

## 🧼 Preprocessing Steps

- Convert stress_level → categorical label  
- Label encode: employee_id, department  
- Standardize numerical features  
- Train-test split (80/20)

---

## 🤖 Model 1 — MLP

A neural network that processes numerical features:
- Dense(32, relu)
- Dense(16, relu)
- Dense(3, softmax)

Evaluations:
- Accuracy
- Classification Report
- Confusion Matrix

---

## 🤝 Model 2 — Neural Collaborative Filtering (NCF)

Embedding-based model using:
- User (employee)
- Item (department)

Layers:
- Embedding for employee_id
- Embedding for department
- Concatenate
- Dense layers → softmax output

Evaluations:
- Accuracy
- Confusion Matrix
- Classification Report

---

## 📊 Results

Both models output:
- Accuracy Score
- Precision, Recall, F1-score
- Confusion Matrix Plot

Performance varies depending on dataset characteristics.

---

## 🔧 How to Run

1. Activate virtual environment:
```bash
source evaluasi/bin/activate
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

3. Launch notebook:
```bash
jupyter notebook
```

Then open:
`stress_prediction_models.ipynb`

---

## 📦 Dependencies

- Python 3.10+
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Matplotlib
- Seaborn

---

## ⚠️ Notes

If running in WSL or CPU-only environment, TensorFlow may show warnings like:

```
Could not find CUDA drivers on your machine, GPU will not be used.
```

These warnings are safe and do not affect model training.

