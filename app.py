"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : app.py
Purpose : Main Streamlit Application
==========================================================
"""

import streamlit as st

from config.settings import (
    APP_TITLE,
    APP_ICON,
    LAYOUT,
    SIDEBAR_STATE
)

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE
)
# -------------------------------------------------------
# Sidebar Custom Content
# -------------------------------------------------------

with st.sidebar:

    st.title("🛡️ Toxicity Detection")

    st.markdown("---")

    st.caption("Developed using NLP + Deep Learning")

# -------------------------------------------------------
# Main Page
# -------------------------------------------------------

st.title(f"{APP_ICON} {APP_TITLE}")

st.markdown("---")

st.markdown(
"""
### Project Overview

This application detects whether an online comment is:

- 🟢 Non-Toxic
- 🔴 Toxic

using Natural Language Processing (NLP) and a Bidirectional LSTM (BiLSTM) Deep Learning model.
"""
)

st.markdown("---")

st.subheader("⚙️ Prediction Pipeline")

st.write("""
1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Tokenization
5. Sequence Padding
6. BiLSTM Model Prediction
7. Toxicity Classification
""")

st.markdown("---")

st.subheader("Technologies Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Language", "Python")

with col2:
    st.metric("Framework", "TensorFlow")

with col3:
    st.metric("UI", "Streamlit")

st.markdown("---")

st.subheader("🧠 BiLSTM Prediction Pipeline")

st.code("""
Raw Comments
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Padding
      │
      ▼
Embedding Layer
      │
      ▼
BiLSTM
      │
      ▼
Sigmoid
      │
      ▼
Prediction
""")

st.markdown("---")

st.caption(
    "Comment Toxicity Detection using NLP & BiLSTM | Developed with Python, TensorFlow, and Streamlit"
)

st.markdown("---")

st.success("✅ Model Loaded Successfully")

st.info(
    "Use the pages in the sidebar to predict comments, upload CSV files, and explore model performance."
)