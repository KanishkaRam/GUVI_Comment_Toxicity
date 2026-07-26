"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : 3_📁_Bulk_Prediction.py
Purpose : Bulk Prediction using CSV Upload
==========================================================
"""

import pandas as pd
import streamlit as st

from utils.predictor import predict_comment
from utils.helper import dataframe_to_csv

from config.settings import (
    APP_TITLE,
    TEXT_COLUMN
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_TITLE} - Bulk Prediction",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Bulk CSV Prediction")

st.markdown("---")

st.write("""
Upload a CSV file containing a **comment_text** column.
The application will predict whether each comment is **Toxic** or **Non-Toxic**.
""")

st.subheader("📄 Sample CSV")

sample_df = pd.DataFrame({
    TEXT_COLUMN: [
        "I love this movie",
        "You are stupid",
        "Have a nice day"
    ]
})

sample_csv = dataframe_to_csv(sample_df)

st.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv,
    file_name="sample_comments.csv",
    mime="text/csv"
)

st.markdown("---")

# -------------------------------------------------------
# Upload CSV
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read CSV.\n\n{e}")
        st.stop()

    # -------------------------------------------------------
    # Validate Column
    # -------------------------------------------------------

    if TEXT_COLUMN not in df.columns:

        st.error(
            f"Column '{TEXT_COLUMN}' not found in uploaded CSV."
        )

        st.stop()

    st.success("CSV Uploaded Successfully")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------------------------------------
    # Start Prediction
    # -------------------------------------------------------

    if st.button("🚀 Predict Entire CSV", use_container_width=True):

        progress = st.progress(0)

        status = st.empty()

        results = []

        total_rows = len(df)

        for index, comment in enumerate(df[TEXT_COLUMN]):

            result = predict_comment(comment)

            results.append(result)

            progress.progress((index + 1) / total_rows)

            status.text(
                f"Processing {index+1} of {total_rows}"
            )

        # -------------------------------------------------------
        # Add Results
        # -------------------------------------------------------

        df["Prediction"] = [
            r["label"]
            for r in results
        ]

        df["Probability"] = [
            r["probability"]
            for r in results
        ]

        df["Confidence"] = [
            r["confidence"]
            for r in results
        ]

        df["Cleaned_Text"] = [
            r["cleaned_text"]
            for r in results
        ]

        progress.empty()

        status.empty()

        st.success("Prediction Completed Successfully")

        st.markdown("---")

        # -------------------------------------------------------
        # Summary
        # -------------------------------------------------------

        toxic = (df["Prediction"] == "Toxic").sum()
        non_toxic = (df["Prediction"] == "Non-Toxic").sum()
        total = len(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Comments", total)

        col2.metric("Toxic",f"{toxic} ({toxic/total:.1%})")

        col3.metric("Non-Toxic",f"{non_toxic} ({non_toxic/total:.1%})")

        st.markdown("---")

        st.subheader("Prediction Results")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.markdown("---")

        csv = dataframe_to_csv(df)

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="bulk_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")

st.caption(
    "Comment Toxicity Detection | Bulk CSV Prediction | Powered by NLP, TensorFlow & BiLSTM"
)