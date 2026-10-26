"""
gradio_app.py - Production Gradio Web Application for Fake News Detection
Supports real-time inference, batch CSV classification, and model metrics.
"""

import os
import sys
import model_utils
from model_utils import (
    SAMPLE_NEWS,
    BENCHMARK_METRICS,
    analyze_linguistic_signals,
    predict_offline_heuristic,
    TransformerNewsClassifier
)

# Global model instance
CURRENT_CLASSIFIER = None
DEFAULT_MODEL_NAME = "dhruvpal/fake-news-bert"


def classify_news_text(text: str, model_source: str, threshold: float = 0.5):
    """
    Classify a single news headline or article.
    Returns:
    - Probabilities dictionary for Gradio Label component
    - Status message / Verdict markdown
    - Sensationalism score markdown
    - Detected linguistic signals
    """
    global CURRENT_CLASSIFIER

    if not text or not text.strip():
        return (
            {"FAKE": 0.0, "REAL": 0.0},
            "⚠️ **Warning**: Please enter a headline or article text.",
            "N/A",
            "N/A"
        )

    words = text.strip().split()
    if len(words) < 4:
        return (
            {"FAKE": 0.0, "REAL": 0.0},
            "⚠️ **Warning**: Text is too short. Please enter at least 5 words.",
            "N/A",
            "N/A"
        )

    # Determine classification engine
    label = "REAL"
    confidence = 0.5
    probs = {"FAKE": 0.5, "REAL": 0.5}
    engine_used = "Offline NLP Baseline"

    if "Offline" in model_source:
        label, confidence, probs = predict_offline_heuristic(text)
    else:
        model_name = DEFAULT_MODEL_NAME
        if "ISOT" in model_source:
            model_name = "prabhatraja445/fake-news-misinformation-distilbert-512"

        try:
            if CURRENT_CLASSIFIER is None or CURRENT_CLASSIFIER.model_name_or_path != model_name:
                CURRENT_CLASSIFIER = TransformerNewsClassifier(model_name)
                CURRENT_CLASSIFIER.load()

            label, confidence, probs = CURRENT_CLASSIFIER.predict(text)
            engine_used = f"Transformer ({model_name})"
        except Exception as e:
            label, confidence, probs = predict_offline_heuristic(text)
            engine_used = f"Offline Baseline (Fallback: {str(e)[:60]}...)"

    # Apply threshold
    prob_fake = probs.get("FAKE", 0.0)
    if prob_fake >= threshold:
        verdict = f"🚨 **PREDICTED AS FAKE NEWS** (Confidence: {prob_fake * 100:.1f}%)"
    else:
        prob_real = probs.get("REAL", 1.0 - prob_fake)
        verdict = f"✅ **PREDICTED AS REAL NEWS** (Confidence: {prob_real * 100:.1f}%)"

    # Analyze linguistic markers
    signals = analyze_linguistic_signals(text)
    sensational_summary = (
        f"**Sensationalism Score:** {signals['sensationalism_score']}/100\n\n"
        f"**Uppercase Ratio:** {signals['uppercase_ratio']}%\n\n"
        f"**Exclamation Marks:** {signals['exclamation_count']}"
    )

    clickbaits = ", ".join(signals["detected_clickbait"]) if signals["detected_clickbait"] else "None detected"
    credibles = ", ".join(signals["detected_credible"]) if signals["detected_credible"] else "None detected"

    signals_summary = (
        f"**Clickbait Words:** {clickbaits}\n\n"
        f"**Credibility Indicators:** {credibles}\n\n"
        f"**Engine Used:** {engine_used}"
    )

    return probs, verdict, sensational_summary, signals_summary


def batch_classify_file(file_obj):
    """
    Batch classification for uploaded CSV or TXT file in Gradio.
    """
    if file_obj is None:
        return None, "Please upload a valid CSV or TXT file."

    import pandas as pd
    file_path = file_obj.name
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            df = pd.DataFrame({"text": lines})

        # Locate text column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            return None, "No text columns found in the dataset."
        col = text_cols[0]

        predictions = []
        confidences = []
        fake_probabilities = []

        for _, row in df.iterrows():
            txt = str(row[col])
            lbl, conf, p = predict_offline_heuristic(txt)
            predictions.append(lbl)
            confidences.append(conf)
            fake_probabilities.append(p.get("FAKE", 0.0))

        df["Prediction"] = predictions
        df["Confidence"] = confidences
        df["Fake_Probability"] = fake_probabilities

        out_path = "batch_classification_results.csv"
        df.to_csv(out_path, index=False)

        summary = (
            f"Successfully classified **{len(df)} articles**!\n\n"
            f"- **Fake Articles:** {(df['Prediction'] == 'FAKE').sum()}\n"
            f"- **Real Articles:** {(df['Prediction'] == 'REAL').sum()}"
        )
        return out_path, summary
    except Exception as e:
        return None, f"Error processing file: {e}"


def get_metrics_table():
    """
    Generate comparison table for metrics tab.
    """
    import pandas as pd
    rows = []
    for m_name, m_val in BENCHMARK_METRICS.items():
        rows.append([
            m_name,
            f"{m_val['accuracy']*100:.2f}%",
            f"{m_val['precision']*100:.2f}%",
            f"{m_val['recall']*100:.2f}%",
            f"{m_val['f1']*100:.2f}%",
            f"{m_val['latency_ms']} ms",
            f"{m_val['params_million']}M"
        ])
    return pd.DataFrame(
        rows,
        columns=["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Latency", "Parameters"]
    )


def create_gradio_app():
    """
    Construct Gradio Blocks user interface.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed in the current environment.")
        print("Install it with: pip install gradio")
        return None

    with gr.Blocks(title="Fake News Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📰 Fake News Detection System (DistilBERT / BERT)\n"
            "State-of-the-Art NLP Misinformation Veracity Classifier with Metrics & Interpretability"
        )

        with gr.Tab("🔍 Single Article Detection"):
            with gr.Row():
                with gr.Column(scale=3):
                    input_text = gr.Textbox(
                        label="News Headline or Full Article Text",
                        placeholder="Paste article, headline, or excerpt here...",
                        lines=8
                    )
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "DistilBERT (dhruvpal/fake-news-bert)",
                                "DistilBERT ISOT (prabhatraja445/fake-news-misinformation-distilbert-512)",
                                "Offline NLP Baseline (Lexical & Heuristic)"
                            ],
                            value="DistilBERT (dhruvpal/fake-news-bert)",
                            label="Model Architecture"
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.5,
                            step=0.05,
                            label="Fake News Sensitivity Threshold"
                        )

                    with gr.Row():
                        btn_clear = gr.Button("Clear", variant="secondary")
                        btn_submit = gr.Button("🚀 Analyze Veracity", variant="primary")

                    gr.Markdown("### 🧪 Quick Examples (Click to Test)")
                    example_list = [[v, "DistilBERT (dhruvpal/fake-news-bert)", 0.5] for v in SAMPLE_NEWS.values()]
                    gr.Examples(
                        examples=example_list,
                        inputs=[input_text, model_dropdown, threshold_slider],
                        label="Click any example to populate"
                    )

                with gr.Column(scale=2):
                    output_verdict = gr.Markdown("### Verdict will appear here")
                    output_label = gr.Label(label="Veracity Class Probabilities", num_top_classes=2)
                    output_sensationalism = gr.Markdown("### Linguistic Signals")
                    output_signals = gr.Markdown("### Keyword Analysis")

            btn_submit.click(
                fn=classify_news_text,
                inputs=[input_text, model_dropdown, threshold_slider],
                outputs=[output_label, output_verdict, output_sensationalism, output_signals]
            )
            btn_clear.click(
                fn=lambda: ("", {"FAKE": 0.0, "REAL": 0.0}, "", "", ""),
                inputs=[],
                outputs=[input_text, output_label, output_verdict, output_sensationalism, output_signals]
            )

        with gr.Tab("📁 Batch File Detection"):
            gr.Markdown("### Upload CSV or TXT file to classify multiple news articles")
            file_input = gr.File(label="Upload News Dataset (.csv or .txt)")
            btn_batch = gr.Button("Run Batch Classification", variant="primary")
            batch_status = gr.Markdown("")
            file_output = gr.File(label="Download Enriched CSV Results")

            btn_batch.click(
                fn=batch_classify_file,
                inputs=[file_input],
                outputs=[file_output, batch_status]
            )

        with gr.Tab("📊 Evaluation Metrics & Benchmarks"):
            gr.Markdown("### Empirical Benchmark Results on the ISOT Fake News Dataset")
            gr.DataFrame(value=get_metrics_table(), interactive=False)
            gr.Markdown("""
            ### 📌 Key Metric Definitions:
            - **Accuracy:** Proportion of all news articles correctly predicted as Real or Fake.
            - **Precision (Fake):** When the model alerts that an article is Fake, how often is it actually Fake?
            - **Recall (Fake):** What percentage of all actual misinformation in the wild is caught?
            - **F1-Score:** The harmonic mean of Precision and Recall ($2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$).
            """)

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    if app:
        print("Launching Gradio App...")
        app.launch(share=False)
