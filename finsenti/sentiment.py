"""
Module for sentiment analysis of news articles.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("soleimanian/financial-roberta-large-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("soleimanian/financial-roberta-large-sentiment")

sentiment_labels = ["neutral", "positive", "negative"]

def chunk_text(text: str, max_tokens: int = 512) -> list:
    """
    Chunk text into smaller segments to fit model input size.

    Parameters:
    text (str): The text to be chunked.
    max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    list: List of text chunks.
    """

    if not text or not text.strip():
        return []

    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def num_tokens(text: str) -> int:
    """
    Calculate the number of tokens in the given text.

    Parameters:
    text (str): The text to be tokenized.

    Returns:
    int: Number of tokens.
    """
    if not text or not text.strip():
        return 0
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def predict_sentiment(text: str) -> tuple:
    """
    Predict sentiment of the given text.

    Parameters:
    text (str): The text for sentiment analysis.

    Returns:
    tuple: Sentiment label and confidence score.
    """

    if not text or not text.strip():
        return "neutral", 0.0

    chunks = chunk_text(text)
    # Calculate number of tokens for each chunk
    token_counts = [max(1, num_tokens(c)) for c in chunks]
    prob_list = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.device.type != "cpu":
                logits = logits.detach().cpu()

            probs = torch.softmax(logits, dim=-1).numpy()[0]
            prob_list.append(probs.tolist())

    # Calculate weighted average of probabilities based on token counts
    token_counts = np.asarray(token_counts, dtype=np.float64)
    prob_arr = np.asarray(prob_list, dtype=np.float64)

    total = token_counts.sum()
    if total <= 0:
        weights = np.full_like(token_counts, 1.0 / len(token_counts))
    else:
        weights = token_counts / total

    weighted_probs = (prob_arr * weights[:, None]).sum(axis=0)

    if hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        sentiment_idx = int(np.argmax(weighted_probs))
        label = id2label[sentiment_idx]
    else:
        # Fallback to your predefined list; ensure its order matches model training
        sentiment_idx = int(np.argmax(weighted_probs))
        label = sentiment_labels[sentiment_idx]

    confidence = float(weighted_probs[sentiment_idx])

    return label, confidence