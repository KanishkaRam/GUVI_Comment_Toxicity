"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : load_model.py
Purpose : Load trained model and artifacts
==========================================================
"""

import pickle
import streamlit as st

from tensorflow.keras.models import load_model

from config.settings import (
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH
)

# -------------------------------------------------------
# Load Deep Learning Model
# -------------------------------------------------------

@st.cache_resource
def load_trained_model() -> object:
    """
    Load the trained BiLSTM model.
    """

    try:
        model = load_model(MODEL_PATH)
        return model

    except FileNotFoundError:
        st.error(f"❌ Model file not found:\n{MODEL_PATH}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Unable to load model.\n\n{e}")
        st.stop()


# -------------------------------------------------------
# Load Tokenizer
# -------------------------------------------------------

@st.cache_resource
def load_tokenizer() -> object:
    """
    Load tokenizer.
    """

    try:
        with open(TOKENIZER_PATH, "rb") as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        return tokenizer

    except FileNotFoundError:
        st.error(f"❌ Tokenizer not found:\n{TOKENIZER_PATH}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Unable to load tokenizer.\n\n{e}")
        st.stop()


# -------------------------------------------------------
# Load Configuration
# -------------------------------------------------------

@st.cache_resource
def load_config() -> dict:
    """
    Load saved configuration.
    """

    try:
        with open(CONFIG_PATH, "rb") as config_file:
            config = pickle.load(config_file)

        return config

    except FileNotFoundError:
        st.error(f"❌ Config file not found:\n{CONFIG_PATH}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Unable to load configuration.\n\n{e}")
        st.stop()


# -------------------------------------------------------
# Load Everything
# -------------------------------------------------------

@st.cache_resource
def load_all() -> tuple:
    

    model = load_trained_model()

    tokenizer = load_tokenizer()

    config = load_config()

    return model, tokenizer, config