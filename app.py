import gradio as gr
from model import predict

# ── Model performance constants (your actual results) ────────────────────────
LR_ACCURACY  = "68.5%"
RIDGE_R2     = "0.235"
CHI2_STAT    = "139.52"
CHI2_PVALUE  = "5.05 × 10⁻³¹"
DATASET_SIZE = "979 daily observations (40,827 deduplicated articles)"

# ── Example inputs ────────────────────────────────────────────────────────────
EXAMPLES = [
    ["RBI holds repo rate steady amid global uncertainty; banking sector "
     "credit growth remains strong at 16% YoY in Q3 FY2024."],
    ["HDFC Bank reports 18% rise in net profit, beats analyst estimates; "
     "Sensex rallies 500 points on strong earnings season across banking sector."],
    ["Rising crude oil prices and weak rupee dampen investor sentiment; "
     "banking stocks under pressure ahead of key Fed rate decision this week."],
    ["SBI announces major expansion of retail loan portfolio targeting MSMEs; "
     "analysts upgrade banking sector outlook to overweight for next quarter."],
    ["RBI imposes penalty on three major private banks for KYC violations; "
     "markets cautious as inflation data comes in higher than expected."],
]

# ── Prediction function ───────────────────────────────────────────────────────
def run_prediction(text):
    if not text or not text.strip():
        return ("—", "—", "—", "—", "—")

    try:
        r = predict(text)

        # Sentiment output
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        sentiment_out = (
            f"{emoji.get(r['sentiment'], '⚪')} "
            f"{r['sentiment'].upper()} "
            f"(FinBERT score: {r['avg_score']})"
        )

        # Direction output
        dir_emoji = {"positive": "📈", "negative": "📉"}
        dir_label = r["direction"]
        direction_out = (
            f"{dir_emoji.get(dir_label, '➡️')} "
            f"Market likely to move {dir_label.upper()} "
            f"(confidence: {r['confidence']}%)"
        )

        # Return magnitude
        pct = r["predicted_return"]
        sign = "+" if pct > 0 else ""
        return_out = f"{sign}{pct}% expected return"

        # Chunk info
        chunks_out = (
            f"{r['chunks_processed']} chunk(s) processed | "
            f"Chunk labels: {', '.join(r['all_labels'])}"
        )

        # Interpretation
        if r["sentiment"] == "positive" and dir_label == "positive":
            interp = "✅ Strong bullish signal — news sentiment and predicted direction align."
        elif r["sentiment"] == "negative" and dir_label == "negative":
            interp = "⚠️ Strong bearish signal — news sentiment and predicted direction align."
        elif r["sentiment"] != dir_label and dir_label != "neutral":
            interp = "⚡ Mixed signal — sentiment and predicted direction diverge. Interpret with caution."
        else:
            interp = "➡️ Neutral signal — limited directional information in the news."

        return sentiment_out, direction_out, return_out, interp, chunks_out

    except Exception as e:
        return (f"Error: {str(e)}", "—", "—", "—", "—")


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Bank NIFTY Sentiment Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
    # 📊 Bank NIFTY Market Direction Predictor
    Paste financial news articles to get a **sentiment-driven market prediction** 
    for the Indian banking sector index (Bank NIFTY).

    **Pipeline:** FinBERT Sentiment → Logistic Regression (direction) + Ridge Regression (return magnitude)
    """)

    # ── Input ─────────────────────────────────────────────────────────────────
    with gr.Row():
        txt = gr.Textbox(
            lines=7,
            placeholder="Paste one or more financial news articles here...\n\nTip: longer articles give more reliable predictions as FinBERT processes them in chunks.",
            label="📰 Financial News Articles",
        )

    btn = gr.Button("🔍 Predict Market Signal", variant="primary", size="lg")

    # ── Outputs ───────────────────────────────────────────────────────────────
    gr.Markdown("### Results")

    with gr.Row():
        sentiment_out  = gr.Textbox(label="📌 News Sentiment (FinBERT)")
        direction_out  = gr.Textbox(label="📈 Predicted Market Direction (Logistic Regression)")

    with gr.Row():
        return_out     = gr.Textbox(label="💹 Predicted Return Magnitude (Ridge Regression)")
        interp_out     = gr.Textbox(label="🧠 Signal Interpretation")

    chunks_out = gr.Textbox(label="⚙️ Processing Info", scale=1)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Markdown("### Try an example")
    gr.Examples(
        examples=EXAMPLES,
        inputs=txt,
        label="Click any example to load it"
    )

    # ── Button click ──────────────────────────────────────────────────────────
    btn.click(
        fn=run_prediction,
        inputs=txt,
        outputs=[sentiment_out, direction_out, return_out, interp_out, chunks_out]
    )

    # ── Model info ────────────────────────────────────────────────────────────
    gr.Markdown(f"""
    ---
    ### 📋 How to Interpret Results
    | Output | Model | What it means |
    |---|---|---|
    | News Sentiment | FinBERT | Positive / negative / neutral tone of the news |
    | Market Direction | Logistic Regression | Predicted up or down movement |
    | Return Magnitude | Ridge Regression | Estimated % return for the day |
    | Signal Interpretation | Rule-based | Whether sentiment and direction agree |

    ### 📊 Model Performance
    | Model | Metric | Value |
    |---|---|---|
    | Chi-Square Test | Statistic / p-value | {CHI2_STAT} / {CHI2_PVALUE} |
    | Logistic Regression | Accuracy | {LR_ACCURACY} |
    | Ridge Regression | R² Score | {RIDGE_R2} |
    | Dataset | Size | {DATASET_SIZE} |

    ### ⚠️ Disclaimer
    This tool is for research and educational purposes only. 
    Predictions are based on news sentiment and should not be used as financial advice.
    """)

demo.launch()
