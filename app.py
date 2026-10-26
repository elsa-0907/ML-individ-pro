"""
app.py - Minimalist, Beautiful Fake News Detector
Single-page, centered layout focused purely on real-time fake news detection.
"""

import time
import streamlit as st
import model_utils
from model_utils import (
    clean_news_text,
    analyze_linguistic_signals,
    predict_offline_heuristic,
    TransformerNewsClassifier
)
import fact_checker
from fact_checker import check_factual_veracity

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Clean, Modern CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide sidebar completely and clean top padding */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    .block-container {
        max-width: 780px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Modern Hero Header */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }

    /* Result Cards */
    .card-real {
        background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.12);
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .card-fake {
        background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.12);
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .verdict-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .verdict-desc {
        font-size: 1rem;
        color: #334155;
        margin-bottom: 0.6rem;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 700;
        background: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Tag pills */
    .signal-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem;
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    .signal-pill-alert {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem;
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Fast Transformer Model Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load pre-trained DistilBERT fake news classifier with fallback."""
    try:
        clf = TransformerNewsClassifier("dhruvpal/fake-news-bert")
        clf.load()
        return clf
    except Exception:
        # Graceful fallback to heuristic engine if torch/network unavailable
        return None


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Instantly verify the authenticity of any news headline or article excerpt using fine-tuned NLP transformers.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Quick Sample Selectors
# -----------------------------------------------------------------------------
SAMPLES = {
    "🚨 Sensational Claim": (
        "BREAKING: Secret documents leaked from a classified underground facility PROVE that "
        "world leaders and mainstream media executives are hiding a miraculous energy cure! "
        "Share this article immediately before government censors delete this page forever!!"
    ),
    "📰 Verified News": (
        "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest "
        "rate at current levels, citing sustained employment gains and moderate inflation figures. "
        "Chairman Jerome Powell stated future adjustments will depend on upcoming economic indicators."
    ),
    "💊 Miracle Cure": (
        "SHOCKING DISCOVERY: Renowned holistic doctor reveals one simple kitchen spice that "
        "completely eradicates diabetes and cancer in just 48 hours without any pills! "
        "Big Pharma is furiously attempting to ban this video. Watch now before it is deleted!"
    )
}

st.write("**Try a sample:**")
sample_cols = st.columns(3)
for idx, (label, sample_text) in enumerate(SAMPLES.items()):
    with sample_cols[idx]:
        if st.button(label, use_container_width=True):
            st.session_state["news_input"] = sample_text

# Manage state for text area
if "news_input" not in st.session_state:
    st.session_state["news_input"] = ""

user_text = st.text_area(
    label="News text:",
    value=st.session_state["news_input"],
    height=160,
    placeholder="Paste news headline, article excerpt, or tweet here...",
    label_visibility="collapsed"
)

# Word & Char Count
char_cnt = len(user_text)
word_cnt = len(user_text.split())

col_actions = st.columns([2, 1])
with col_actions[0]:
    analyze_btn = st.button("🔍 Check Veracity", type="primary", use_container_width=True)
with col_actions[1]:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["news_input"] = ""
        st.rerun()

# -----------------------------------------------------------------------------
# Prediction & Results Display
# -----------------------------------------------------------------------------
if analyze_btn:
    if not user_text.strip():
        st.warning("Please paste or type some news text first.")
    elif word_cnt < 3:
        st.warning("Text is too short. Please enter a complete headline or claim to analyze.")
    else:
        with st.spinner("Analyzing text patterns and verifying facts..."):
            time.sleep(0.15)  # smooth visual feedback
            
            # 1. Fact-check verifiable semantic contradictions & hoaxes
            factual_check = check_factual_veracity(user_text)
            if factual_check:
                is_false, explanation = factual_check
                verdict = "FAKE"
                confidence = 0.99
                probs = {"FAKE": 0.99, "REAL": 0.01}
                fact_check_reason = explanation
            else:
                fact_check_reason = None
                clf = load_classifier()
                if clf is not None:
                    try:
                        verdict, confidence, probs = clf.predict(user_text)
                    except Exception:
                        verdict, confidence, probs = predict_offline_heuristic(user_text)
                else:
                    verdict, confidence, probs = predict_offline_heuristic(user_text)

            signals = analyze_linguistic_signals(user_text)
            prob_fake = probs.get("FAKE", 0.0)
            prob_real = probs.get("REAL", 0.0)

        # Result Display Card
        if verdict == "REAL":
            st.markdown(f"""
            <div class="card-real">
                <div class="verdict-title" style="color: #15803D;">✅ Real News</div>
                <div class="verdict-desc">
                    This article uses objective phrasing and neutral language consistent with reliable reporting.
                </div>
                <div class="confidence-badge" style="color: #15803D; border: 1px solid #86EFAC;">
                    Confidence: <b>{confidence * 100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            reason_html = ""
            if fact_check_reason:
                reason_html = f"""
                <div style="margin-top: 0.8rem; padding: 0.7rem 1rem; background: #FFF1F2; border-radius: 8px; border: 1px solid #FECDD3; color: #9F1239; font-size: 0.95rem; text-align: left;">
                    <b>💡 Fact-Check Detection:</b> {fact_check_reason}
                </div>
                """
            st.markdown(f"""
            <div class="card-fake">
                <div class="verdict-title" style="color: #B91C1C;">🚨 Potential Fake News / False Claim</div>
                <div class="verdict-desc">
                    This claim contains verifiable factual contradictions, sensationalism, or clickbait framing.
                </div>
                <div class="confidence-badge" style="color: #B91C1C; border: 1px solid #FCA5A5;">
                    Confidence: <b>{confidence * 100:.1f}%</b>
                </div>
                {reason_html}
            </div>
            """, unsafe_allow_html=True)

        # Probabilities & Indicators
        st.write("")
        st.write("#### Authenticity Breakdown")
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.write(f"**Real Probability:** {prob_real * 100:.1f}%")
            st.progress(float(prob_real))
        with col_prob2:
            st.write(f"**Fake Probability:** {prob_fake * 100:.1f}%")
            st.progress(float(prob_fake))

        # Linguistic Flags (if any found)
        st.write("")
        st.write("#### Linguistic Signals")
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Sensationalism", f"{signals['sensationalism_score']}/100")
        info_col2.metric("All-Caps Ratio", f"{signals['uppercase_ratio']}%")
        info_col3.metric("Exclamations", signals["exclamation_count"])

        if signals["detected_clickbait"]:
            st.write("**Clickbait Trigger Words Found:**")
            pills_html = " ".join([f'<span class="signal-pill-alert">{w}</span>' for w in signals["detected_clickbait"]])
            st.markdown(pills_html, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "<center style='color:#94A3B8;'>Powered by Fine-Tuned DistilBERT NLP • Always cross-reference breaking news with verified sources.</center>",
    unsafe_allow_html=True
)
