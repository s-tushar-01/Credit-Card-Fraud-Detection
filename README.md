# 💳 Credit Card Fraud Detection System

A machine learning project that detects fraudulent credit card transactions using classification models.  
The system handles highly imbalanced transaction data and evaluates performance using precision, recall, F1-score, and ROC-AUC metrics.

---

## 📊 Problem Statement

Credit card fraud is a major financial risk. The goal of this project is to build a classification model that can accurately identify fraudulent transactions while minimizing false positives.

---

## 📁 Dataset

- Source: Kaggle - Credit Card Fraud Detection Dataset
- Total Transactions: 284,807
- Fraud Cases: 492
- Highly imbalanced dataset

Features include:
- Time
- Amount
- V1–V28 (PCA-transformed features)
- Class (0 = Normal, 1 = Fraud)

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Machine Learning Models Used

1. Logistic Regression (with class_weight='balanced')
2. Random Forest Classifier

---

## ⚙️ Project Structure
CreditCardFraudDetection/
│
├── data/
│ └── creditcard.csv
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md


---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt


3. Run the project:


python main.py


---

## 📈 Evaluation Metrics

Since the dataset is highly imbalanced, accuracy alone is not sufficient.

The following metrics are used:

- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🎯 Key Learnings

- Handling imbalanced datasets using class weights
- Model evaluation using appropriate metrics
- Modular project structure for scalable ML systems
- Comparing performance of multiple classification models

---

## 📌 Future Improvements

- SMOTE for oversampling
- Hyperparameter tuning
- XGBoost implementation
- Streamlit deployment for real-time fraud prediction

---

## 👨‍💻 Author

Tushar  
Machine Learning & AI Enthusiast