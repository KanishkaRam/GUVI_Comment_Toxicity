"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : 1_📊_Dashboard.py
Purpose : Dataset and Model Dashboard
==========================================================
"""

import os
import pickle

import pandas as pd
import streamlit as st

from config.settings import (
    APP_TITLE,
    PROCESSED_DATA_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    TEXT_COLUMN,
    TARGET_COLUMN
)

from utils.visualization import (
    plot_class_distribution,
    plot_training_history,
    generate_wordcloud,
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_TITLE} - Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard")

st.markdown("---")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found:\n{PROCESSED_DATA_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Unable to load dataset.\n\n{e}")
        st.stop()

df = load_dataset()


# -------------------------------------------------------
# Dataset Statistics
# -------------------------------------------------------

st.subheader("📂 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Toxic Comments", int(df[TARGET_COLUMN].sum()))
col4.metric(
    "Non-Toxic",
    int((df[TARGET_COLUMN] == 0).sum())
)

st.markdown("---")

# -------------------------------------------------------
# Preview Dataset
# -------------------------------------------------------

st.subheader("📄 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------
# Class Distribution
# -------------------------------------------------------

st.subheader("📊 Class Distribution")

fig = plot_class_distribution(
    df,
    TARGET_COLUMN
)

st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------
# Word Cloud
# -------------------------------------------------------

st.subheader("☁️ Most Frequent Words")

toxic_text = " ".join(
    df[df[TARGET_COLUMN] == 1][TEXT_COLUMN].astype(str)
)

non_toxic_text = " ".join(
    df[df[TARGET_COLUMN] == 0][TEXT_COLUMN].astype(str)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Toxic Comments")
    st.pyplot(generate_wordcloud(toxic_text))

with col2:
    st.subheader("🟢 Non-Toxic Comments")
    st.pyplot(generate_wordcloud(non_toxic_text))

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
    st.warning("Training history not found.")

st.markdown("---")

# -------------------------------------------------------
# Model Metrics
# -------------------------------------------------------

st.subheader("📋 Model Metrics")

if os.path.exists(METRICS_PATH):

    metrics = pd.read_csv(METRICS_PATH)

    st.dataframe(
        metrics,
        use_container_width=True
    )

else:
    st.warning("Metrics file not found.")

st.markdown("---")

st.success("Dashboard Loaded Successfully")