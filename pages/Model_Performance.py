"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : 4_📈_Model_Performance.py
Purpose : Display Model Evaluation Metrics
==========================================================
"""

import os
import pickle
import pandas as pd
import streamlit as st

from config.settings import (
    APP_TITLE,
    HISTORY_PATH,
    METRICS_PATH,
    CONFIG_PATH
)

from utils.visualization import (
    plot_training_history,
    metrics_dataframe
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_TITLE} - Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance")

st.markdown("---")

st.write("""
This page summarizes the performance of the trained
Bidirectional LSTM model on the toxicity classification task.
""")

# -------------------------------------------------------
# Model Metrics
# -------------------------------------------------------

st.subheader("📊 Evaluation Metrics")

if os.path.exists(METRICS_PATH):

    metrics = pd.read_csv(METRICS_PATH)

    st.dataframe(
        metrics,
        use_container_width=True
    )

else:
    st.warning("⚠️ model_metrics.csv not found.")

st.markdown("---")

# -------------------------------------------------------
# Training History
# -------------------------------------------------------

st.subheader("📈 Training History")

if os.path.exists(HISTORY_PATH):

    with open(HISTORY_PATH, "rb") as file:
        history = pickle.load(file)

    fig = plot_training_history(history)

    st.pyplot(fig)

else:
    st.warning("⚠️ Training history not found.")

st.markdown("---")

# -------------------------------------------------------
# Training Configuration
# -------------------------------------------------------

st.subheader("⚙️ Training Configuration")

if os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "rb") as file:
        config = pickle.load(file)

    config_df = pd.DataFrame(
        config.items(),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        config_df,
        use_container_width=True
    )

else:
    st.warning("⚠️ Config file not found.")

st.markdown("---")

# -------------------------------------------------------
# Model Summary
# -------------------------------------------------------

st.subheader("🧠 Model Architecture")

st.code("""
Input Layer
      │
Embedding Layer
      │
Bidirectional LSTM
      │
Dropout
      │
Dense (ReLU)
      │
Dense (Sigmoid)
      │
Prediction
""")

st.markdown("---")

# -------------------------------------------------------
# Model Information
# -------------------------------------------------------

st.subheader("📌 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "BiLSTM")

with col2:
    st.metric("Task", "Binary Classification")

with col3:
    st.metric("Loss Function", "Binary Crossentropy")

st.markdown("---")

st.success("✅ Model evaluation loaded successfully.")

st.caption(
    "Performance Evaluation | NLP + TensorFlow + BiLSTM"
)