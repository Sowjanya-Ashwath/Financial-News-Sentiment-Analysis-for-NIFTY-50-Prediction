import joblib
import torch
import torch.nn.functional as F
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
FINBERT       = "ProsusAI/finbert"
LABELS        = ["negative", "neutral", "positive"]
SENTIMENT_MAP = {"negative": -1, "neutral": 0, "positive": 1}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading FinBERT...")
tokenizer     = AutoTokenizer.from_pretrained(FINBERT)
finbert       = AutoModelForSequenceClassification.from_pretrained(FINBERT)
finbert.eval()

print("Loading ML models...")
clf_model     = joblib.load("logistic_daily_model.pkl")
ridge_model   = joblib.load("ridge_daily_model.pkl")
label_encoder = joblib.load("market_label_encoder.pkl")
print("All models loaded.")

# ── Text cleaning (matches your notebook clean_text) ─────────────────────────
def clean_text(t):
    t = re.sub(r"https?://\S+", "", str(t))
    t = re.sub(r"[^\w\s.,!'-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ── Chunk text (matches your notebook Cell 21 with 50-token overlap) ──────────
def chunk_text(text, max_len=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_len - 50):
        chunk = tokens[i : i + (max_len - 50)]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks if chunks else [text]

# ── Main prediction (matches your notebook Cell 22 logic exactly) ─────────────
def predict(text: str) -> dict:
    text   = clean_text(text)
    chunks = chunk_text(text)

    all_scores = []
    all_labels = []

    for ch in chunks:
        inputs = tokenizer(
            ch, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = finbert(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

        label = LABELS[probs.argmax()]
        conf  = float(probs.max().item())

        # confidence-weighted score — matches your notebook
        if label == "positive":
            score = conf
        elif label == "negative":
            score = -conf
        else:
            score = 0.0

        all_scores.append(score)
        all_labels.append(label)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_score = float(np.mean(all_scores))

    if avg_score > 0.1:
        final_sentiment = "positive"
    elif avg_score < -0.1:
        final_sentiment = "negative"
    else:
        final_sentiment = "neutral"

    # ── Logistic Regression: direction (bucketed -1/0/1) ──────────────────────
    sentiment_num   = SENTIMENT_MAP[final_sentiment]
    direction_num   = clf_model.predict([[sentiment_num]])[0]
    direction_label = label_encoder.inverse_transform([direction_num])[0]
    direction_proba = clf_model.predict_proba([[sentiment_num]])[0]
    confidence      = round(float(max(direction_proba)) * 100, 1)

    # ── Ridge Regression: return magnitude (continuous avg_score) ─────────────
    predicted_return     = float(ridge_model.predict([[avg_score]])[0])
    predicted_return_pct = round(predicted_return * 100, 4)

    return {
        "sentiment":         final_sentiment,
        "avg_score":         round(avg_score, 4),
        "direction":         direction_label,
        "confidence":        confidence,
        "predicted_return":  predicted_return_pct,
        "chunks_processed":  len(chunks),
        "all_labels":        all_labels,
    }
