"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : 2_🔍_Single_Prediction.py
Purpose : Predict toxicity for a single comment
==========================================================
"""

import streamlit as st

from utils.predictor import predict_comment
from utils.helper import (
    prediction_to_dataframe,
    dataframe_to_csv,
    prediction_badge,
    format_probability,
    format_confidence
)

from config.settings import APP_TITLE

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_TITLE} - Single Prediction",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single Comment Prediction")

st.markdown("---")

st.write(
    """
Enter a comment below and click **Predict** to classify it as
**Toxic** or **Non-Toxic**.
"""
)

# -------------------------------------------------------
# Input
# -------------------------------------------------------

comment = st.text_area(
    "Enter Comment",
    height=180,
    placeholder="Type your comment here..."
)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("🚀 Predict", use_container_width=True):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
        st.stop()

    with st.spinner("Predicting..."):

        result = predict_comment(comment)

    st.success("Prediction Completed")

    st.markdown("---")

    if result["label"] == "Toxic":
        st.error("🚨 The comment is predicted as Toxic.")

    elif result["label"] == "Non-Toxic":
        st.success("✅ The comment is predicted as Non-Toxic.")

    elif result["label"] == "Empty Input":
        st.warning("⚪ Empty input detected.")

    elif result["label"] == "No Valid Text":
        st.warning("🟡 No valid words found after preprocessing.")

    else:
        st.error(result["label"])

    # -----------------------------------------
    # Display Result
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Prediction")

        st.metric(
            "Class",
            prediction_badge(result["label"])
        )

        st.metric(
            "Confidence",
            format_confidence(result["confidence"])
        )

        st.metric(
            "Probability",
            format_probability(result["probability"])
        )

    with col2:

        st.subheader("Text Processing")

        st.text_area(
            "Original Comment",
            result["original_text"],
            height=100,
            disabled=True
        )

        st.text_area(
            "Cleaned Comment",
            result["cleaned_text"],
            height=100,
            disabled=True
        )

    st.markdown("---")

    st.subheader("Prediction Details")

    st.dataframe(
        prediction_to_dataframe(result),
        use_container_width=True
    )

    st.markdown("---")
    
    st.subheader("📊 Toxicity Probability")

    toxicity = float(result["probability"])

    st.progress(toxicity)

    st.metric("Toxicity Score", f"{toxicity*100:.2f}%")
    
    # -----------------------------------------
    # Download
    # -----------------------------------------

    csv = dataframe_to_csv(
        prediction_to_dataframe(result)
    )

    st.download_button(
        label="📥 Download Prediction",
        data=csv,
        file_name="prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, TensorFlow, and NLP"
)