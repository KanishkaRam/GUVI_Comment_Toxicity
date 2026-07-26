"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : helper.py
Purpose : Common helper functions
==========================================================
"""

import pandas as pd


# -------------------------------------------------------
# Convert Prediction Dictionary to DataFrame
# -------------------------------------------------------

def prediction_to_dataframe(result: dict) -> pd.DataFrame:
    """
    Convert prediction dictionary into a DataFrame.

    Parameters
    ----------
    result : dict
        Prediction dictionary.

    Returns
    -------
    pd.DataFrame
    """

    return pd.DataFrame([result])


# -------------------------------------------------------
# Convert DataFrame to CSV
# -------------------------------------------------------

def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame into CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bytes
    """

    return df.to_csv(index=False).encode("utf-8")


# -------------------------------------------------------
# Format Probability
# -------------------------------------------------------

def format_probability(probability: float) -> str:
    """
    Format probability.

    Example:
        0.9452 -> 94.52%
    """

    return f"{probability * 100:.2f}%"


# -------------------------------------------------------
# Format Confidence
# -------------------------------------------------------

def format_confidence(confidence: float) -> str:
    """
    Format confidence.

    Example:
        96.84 -> 96.84%
    """

    return f"{confidence:.2f}%"


# -------------------------------------------------------
# Prediction Badge
# -------------------------------------------------------

def prediction_badge(label: str) -> str:
    """
    Return badge for prediction.
    """

    BADGES = {
    "Toxic": "🔴 Toxic",
    "Non-Toxic": "🟢 Non-Toxic",
    "Empty Input": "⚪ Empty Input",
    "No Valid Text": "🟡 No Valid Text"
    }
    return BADGES.get(label, "⚠️ Unknown")