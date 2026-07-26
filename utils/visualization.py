"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : visualization.py
Purpose : Visualization utilities
==========================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


# -------------------------------------------------------
# Plot Training History
# -------------------------------------------------------

def plot_training_history(history: dict):
    """
    Plot training & validation accuracy and loss.

    Parameters
    ----------
    history : dict
        Training history dictionary.
    """

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # Accuracy
    ax[0].plot(history["accuracy"], label="Train")
    ax[0].plot(history["val_accuracy"], label="Validation")
    ax[0].set_title("Model Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss
    ax[1].plot(history["loss"], label="Train")
    ax[1].plot(history["val_loss"], label="Validation")
    ax[1].set_title("Model Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()

    return fig

# -------------------------------------------------------
# Plot Confusion Matrix
# -------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Toxic","Toxic"],
        yticklabels=["Non-Toxic","Toxic"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig

# -------------------------------------------------------
# Class Distribution
# -------------------------------------------------------

def plot_class_distribution(df, target_column):

    counts = df[target_column].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    return fig

# -------------------------------------------------------
# Prediction Probability
# -------------------------------------------------------

def plot_prediction_probability(probability):

    fig, ax = plt.subplots(figsize=(6,1.5))

    ax.barh(
        ["Toxic Probability"],
        [probability],
    )

    ax.set_xlim(0,1)

    ax.set_xlabel("Probability")

    return fig

# -------------------------------------------------------
# Word Cloud
# -------------------------------------------------------

def generate_wordcloud(text):

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.imshow(wc)

    ax.axis("off")

    return fig

# -------------------------------------------------------
# Model Metrics Table
# -------------------------------------------------------

def metrics_dataframe(metrics_dict):

    return pd.DataFrame(
        metrics_dict.items(),
        columns=["Metric","Value"]
    )