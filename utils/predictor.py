"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : predictor.py
Purpose : Prediction utilities
==========================================================
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import preprocess_text
from utils.load_model import load_all

from config.settings import (
    THRESHOLD,
    CLASS_NAMES
)

# -------------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------------

model, tokenizer, config = load_all()

MAX_LEN = config["max_len"]


# -------------------------------------------------------
# Predict Single Comment
# -------------------------------------------------------

def predict_comment(text: str) -> dict:
    """
    Predict toxicity of a single comment.

    Parameters
    ----------
    text : str
        Input comment.

    Returns
-------
dict
    Dictionary containing:
        - original_text
        - cleaned_text
        - probability
        - confidence
        - prediction
        - label
        - threshold
    """
    if text is None or str(text).strip() == "":
        return {
        "original_text": "",
        "cleaned_text": "",
        "probability": 0.0,
        "confidence": 0.0,
        "prediction": 0,
        "label": "Empty Input"
        }
    # ----------------------------
    # Preprocess
    # ----------------------------

    cleaned_text = preprocess_text(text)
    

    if cleaned_text == "":
        return {
            "original_text": text,
            "cleaned_text": "",
            "probability": 0.0,
            "confidence": 0.0,
            "prediction": 0,
            "label": "No Valid Text"
        }

    # ----------------------------
    # Tokenization
    # ----------------------------

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # ----------------------------
    # Prediction
    # ----------------------------

    try:
        probability = float(
            model.predict(
                padded_sequence,
                verbose=0
            )[0][0]
        )

    except Exception as e:
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "probability": 0.0,
            "confidence": 0.0,
            "prediction": 0,
            "label": f"Prediction Error: {str(e)}"
        }
    prediction = int(probability >= THRESHOLD)

    confidence = max(probability, 1 - probability)

    # ----------------------------
    # Return Result
    # ----------------------------

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "probability": round(probability, 4),
        "confidence": round(confidence * 100, 2),
        "prediction": prediction,
        "label": CLASS_NAMES[prediction],
        "threshold": THRESHOLD
    }