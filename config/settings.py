"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : settings.py
Purpose : Global configuration file
==========================================================
"""

import os

# -------------------------------------------------------
# Base Project Directory
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------
# Data Paths
# -------------------------------------------------------

RAW_DATA_PATH = os.path.join(BASE_DIR,"train.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "toxicity_clean.csv")

# -------------------------------------------------------
# Model Artifacts
# -------------------------------------------------------

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR,"final_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACT_DIR,"tokenizer.pkl")
CONFIG_PATH = os.path.join(ARTIFACT_DIR,"config.pkl")
HISTORY_PATH = os.path.join(ARTIFACT_DIR,"history.pkl")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "model_metrics.csv")

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

APP_TITLE = "Comment Toxicity Detection"
APP_ICON = "🛡️"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# -------------------------------------------------------
# Prediction Configuration
# -------------------------------------------------------

THRESHOLD = 0.50
MAX_UPLOAD_SIZE_MB = 20
ALLOWED_FILE_TYPES = ["csv"]

# -------------------------------------------------------
# Theme Colors
# -------------------------------------------------------

PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#FF5722"
SUCCESS_COLOR = "#4CAF50"
WARNING_COLOR = "#FFC107"
ERROR_COLOR = "#F44336"

# -------------------------------------------------------
# Labels
# -------------------------------------------------------

CLASS_NAMES = {
    0: "Non-Toxic",
    1: "Toxic"
}

PREDICTION_LABELS = {
    0: "🟢 Non-Toxic",
    1: "🔴 Toxic"
}

# -------------------------------------------------------
# Deep Learning Parameters
# -------------------------------------------------------

VOCAB_SIZE = 30000
EMBEDDING_DIM = 128
MAX_LEN = 150         
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
OOV_TOKEN = "<OOV>"

# -------------------------------------------------------
# Dataset Columns
# -------------------------------------------------------

TEXT_COLUMN = "comment_text"
TARGET_COLUMN = "target"