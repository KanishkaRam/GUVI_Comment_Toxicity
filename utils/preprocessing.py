"""
==========================================================
Project : Comment Toxicity Detection using NLP & BiLSTM
File    : preprocessing.py
Purpose : Text preprocessing utilities
==========================================================
"""

import re
import string
import contractions
import emoji
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------
# Download NLTK resources (only once)
# -------------------------------------------------------

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# -------------------------------------------------------
# Initialize Objects
# -------------------------------------------------------

STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------
# Individual Cleaning Functions
# -------------------------------------------------------

def lowercase_text(text):
    """Convert text to lowercase."""
    return str(text).lower()


def remove_html(text):
    """Remove HTML tags."""
    return re.sub(r"<.*?>", "", text)


def remove_urls(text):
    """Remove URLs."""
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def expand_contractions(text):
    """Expand contractions."""
    return contractions.fix(text)


def remove_emojis(text):
    """Remove emojis."""
    return emoji.replace_emoji(text, replace="")


def remove_numbers(text):
    """Remove numbers."""
    return re.sub(r"\d+", "", text)


def remove_punctuation(text):
    """Remove punctuation."""
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


def remove_extra_spaces(text):
    """Remove extra spaces."""
    return re.sub(r"\s+", " ", text).strip()

def remove_mentions(text):
    return re.sub(r"@\w+", "", text)

def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def remove_stopwords(text):
    """Remove stopwords."""
    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(filtered_words)


def lemmatize_text(text):
    """Lemmatize words."""
    words = text.split()

    lemmas = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(lemmas)

# -------------------------------------------------------
# Complete Pipeline
# -------------------------------------------------------

def preprocess_text(text):
    """
    Complete preprocessing pipeline.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if text == "":
        return ""

    text = lowercase_text(text)

    text = remove_html(text)

    text = remove_urls(text)

    text = expand_contractions(text)

    text = remove_emojis(text)

    text = remove_mentions(text)

    text = remove_hashtags(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = remove_extra_spaces(text)

    text = remove_stopwords(text)

    text = lemmatize_text(text)

    

    return text