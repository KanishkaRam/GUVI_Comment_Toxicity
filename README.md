# 🛡️ Comment Toxicity Detection using NLP & BiLSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154F7D)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Comment Toxicity Detection is a Natural Language Processing (NLP) application that automatically classifies user comments as **Toxic** or **Non-Toxic** using a **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model.

The project includes a modern **Streamlit web application** that supports:

* 🔹 Single Comment Prediction
* 🔹 Bulk CSV Prediction
* 🔹 Interactive Dashboard
* 🔹 Prediction Confidence Scores
* 🔹 Downloadable Prediction Results

This project demonstrates the complete Machine Learning workflow—from data preprocessing and model training to deployment through an interactive web interface.

---

## 🚀 Features

* ✅ Comment preprocessing using NLP
* ✅ Text cleaning and normalization
* ✅ Tokenization & Padding
* ✅ BiLSTM Deep Learning Model
* ✅ Single Comment Prediction
* ✅ Bulk CSV Prediction
* ✅ Interactive Streamlit UI
* ✅ Prediction Confidence Score
* ✅ CSV Download Support
* ✅ Modular Project Structure

---

## 🛠️ Tech Stack

| Category         | Technologies        |
| ---------------- | ------------------- |
| Language         | Python              |
| Framework        | Streamlit           |
| Deep Learning    | TensorFlow, Keras   |
| NLP              | NLTK                |
| Machine Learning | Scikit-Learn        |
| Data Processing  | Pandas, NumPy       |
| Visualization    | Matplotlib, Seaborn |
| Version Control  | Git & GitHub        |

---

## 📂 Project Structure

```text
CommentToxicity/
│
├── app.py
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py
│
├── model/
│   ├── model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── pages/
│   ├── Home.py
│   ├── Single_Prediction.py
│   ├── Bulk_Prediction.py
│   └── Dashboard.py
│
├── utils/
│   ├── preprocess.py
│   ├── predictor.py
│   └── helper.py
│
├── assets/
│
└── .gitignore
```

---

## 🧠 Machine Learning Workflow

1. Dataset Collection
2. Data Cleaning
3. Text Preprocessing
4. Tokenization
5. Sequence Padding
6. Train-Test Split
7. BiLSTM Model Training
8. Model Evaluation
9. Model Saving
10. Streamlit Deployment

---

## 🔍 Text Preprocessing

The following preprocessing steps are performed before prediction:

* Convert text to lowercase
* Remove HTML tags
* Remove URLs
* Remove punctuation
* Remove numbers
* Remove extra spaces
* Remove stop words
* Lemmatization
* Tokenization
* Sequence Padding

---

## 🤖 Deep Learning Model

**Architecture**

* Embedding Layer
* Bidirectional LSTM Layer
* Dropout Layer
* Dense Layer
* Sigmoid Output Layer

**Loss Function**

```
Binary Crossentropy
```

**Optimizer**

```
Adam
```

**Output**

```
0 → Non-Toxic
1 → Toxic
```

---

## 📊 Streamlit Application

The application provides the following modules:

### 🏠 Home

* Project overview
* Navigation
* Quick instructions

### 💬 Single Prediction

* Predict toxicity for individual comments
* Confidence score

### 📁 Bulk Prediction

* Upload CSV
* Predict multiple comments
* Download predictions

### 📈 Dashboard

* Dataset summary
* Prediction distribution
* Interactive visualizations

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/KanishkaRam/GUVI_Comment_Toxicity.git
```

### Navigate to Project

```bash
cd GUVI_Comment_Toxicity
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📦 Required Libraries

* TensorFlow
* Keras
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib
* Seaborn
* Joblib

---

## 📁 Dataset

The original training datasets are **not included** in this repository because they exceed GitHub's file size limits.

If required, download the dataset separately and place it in the appropriate project directory before training the model.

---

## 📸 Application Preview

Add screenshots here after deployment.

Example:

```
Home Page

Single Prediction

Bulk Prediction

Dashboard
```

---

## 🎯 Future Enhancements

* Multi-label toxicity classification
* Explainable AI (SHAP/LIME)
* REST API using FastAPI
* Docker support
* Cloud deployment
* User authentication
* Real-time moderation
* Model retraining pipeline

---

## 👩‍💻 Author

**Kanishka Thiyagarajan**

AI & Machine Learning Enthusiast

GitHub:
https://github.com/KanishkaRam

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork the repository

🛠️ Contribute with improvements

---


