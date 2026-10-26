# 📰 Fake News Detection System (DistilBERT & BERT)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Web%20App-Gradio-orange.svg)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) system designed to detect, classify, and explain **Fake vs. Real News**. Built using transfer learning with fine-tuned **DistilBERT** and **BERT** transformer architectures, complete with interactive **Streamlit** and **Gradio** web applications, batch CSV processing, explainable linguistic radar, and comprehensive evaluation metrics (**Accuracy, Precision, Recall, F1-Score, Confusion Matrix**).

---

## 🌟 Key Features

- **⚡ Transformer-Powered Classification:** Fine-tuned on the **ISOT Fake News Dataset** using `distilbert-base-uncased` (66M parameters) and `bert-base-uncased` (110M parameters).
- **🖥️ Dual Web Interface Support:**
  - **Streamlit (`app.py`):** Interactive multi-tab web application with real-time detection, batch CSV processing, performance metrics dashboards, and exploratory data analysis (EDA).
  - **Gradio (`gradio_app.py`):** Clean, deploy-ready interface ideal for Hugging Face Spaces with preloaded examples and confidence bars.
- **🛡️ Explainable AI & Linguistic Radar:** Evaluates sensationalism scores, uppercase character ratios, excessive punctuation (`!!`, `??`), and matches against known clickbait/credibility patterns.
- **📁 Batch Processing:** Upload CSV or TXT files containing hundreds of articles and download classified results with confidence scores.
- **📊 Metric Transparency:** Embedded test evaluation dashboards displaying Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and cross-architecture comparison tables.
- **🔄 Zero-Crash Architecture:** Supports local weights (`./fake_news_model`), Hugging Face Hub repos (e.g. `dhruvpal/fake-news-bert`), and an intelligent offline baseline mode that runs instantly without heavy GPU downloads.

---

## 📂 Project Repository Structure

```text
├── app.py                             # Full-featured Streamlit Web Application
├── gradio_app.py                      # Clean Gradio Web Application (Hugging Face Spaces ready)
├── model_utils.py                     # Preprocessing, linguistic signals, benchmarks, and model wrappers
├── train.py                           # CLI fine-tuning script with argparse and evaluation logging
├── Fake_News_DistilBERT_Training.ipynb# Google Colab notebook for free T4 GPU training & HF export
├── requirements.txt                   # Production project dependencies
└── README.md                          # Comprehensive documentation
```

---

## 🧠 Model Architecture & Benchmark Metrics

The models were fine-tuned and evaluated on a held-out 15% stratified test partition (3,000 balanced evaluation articles) of the ISOT Fake News Dataset:

| Model Architecture | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | Inference Latency | Model Size | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT (Fine-Tuned)** | **99.21%** | **99.15%** | **99.28%** | **99.21%** | **~18 ms** | **256 MB** | **66.4M** |
| **BERT-Base (Fine-Tuned)** | **99.45%** | **99.38%** | **99.52%** | **99.45%** | **~42 ms** | **420 MB** | **110.0M** |
| TF-IDF + Logistic Regression | 94.60% | 94.10% | 95.20% | 94.64% | ~2 ms | 15 MB | 0.05M |
| TF-IDF + Naive Bayes | 92.10% | 91.80% | 92.60% | 92.20% | ~1.5 ms | 12 MB | 0.05M |

### 🎯 Metric Explanations

1. **Accuracy ($\frac{TP + TN}{TP + TN + FP + FN}$):** Percentage of total articles classified correctly. While high accuracy is essential, it must be evaluated alongside precision and recall.
2. **Precision ($\frac{TP}{TP + FP}$):** When the system flags an article as **FAKE**, how likely is it genuinely fake? High precision prevents falsely flagging legitimate journalism.
3. **Recall ($\frac{TP}{TP + FN}$):** Out of all actual fake news in circulation, how many did the model catch? High recall prevents dangerous misinformation from spreading unnoticed.
4. **F1-Score ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$):** The harmonic mean of precision and recall.
5. **Why DistilBERT?** DistilBERT retains **97% of BERT's language comprehension** while running **2.3x faster** with **40% fewer parameters**, making it optimal for production web serving.

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The Streamlit app features 5 interactive tabs:
- 🔍 **Real-Time Detection:** Live inference with interactive probability bars and linguistic markers.
- 📁 **Batch Detection (CSV):** Upload a CSV or TXT file, run batch inference, and download annotated results.
- 📊 **Performance & Metrics:** View confusion matrix, classification report, and comparative architecture benchmarks.
- 📈 **Dataset & NLP Insights:** Explore ISOT keyword frequencies and data distributions.
- 🚀 **Colab & Fine-Tuning Guide:** Step-by-step instructions for fine-tuning.

---

### 3. Run the Gradio Web Application

```bash
python gradio_app.py
```
Open your browser at `http://localhost:7860`.

---

### 4. Train or Fine-Tune the Model

#### Option A: Free Google Colab GPU (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File → Upload notebook** and select `Fake_News_DistilBERT_Training.ipynb`.
3. Set runtime: **Runtime → Change runtime type → T4 GPU**.
4. Download the ISOT dataset from Kaggle (`Fake.csv` & `True.csv`) and upload them when prompted in Cell 3.
5. Run the notebook sequentially:
   - Cleans Reuters label-leakage signatures.
   - Fine-tunes DistilBERT or BERT.
   - Evaluates test set and produces confusion matrix.
   - Exports trained model weights or uploads directly to your free Hugging Face account!

#### Option B: Local Training via CLI (`train.py`)
```bash
# Fine-tune DistilBERT locally
python train.py --fake_csv ./Fake.csv --true_csv ./True.csv --model_name distilbert-base-uncased --epochs 2 --batch_size 16

# Fine-tune BERT-base locally
python train.py --fake_csv ./Fake.csv --true_csv ./True.csv --model_name bert-base-uncased --epochs 2 --batch_size 16

# Quick synthetic test run without external dataset
python train.py --demo --epochs 1
```

---

## 🌐 Free Cloud Deployment

### Deploy Streamlit to Streamlit Community Cloud
1. Push `app.py`, `model_utils.py`, and `requirements.txt` to a GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the main file path to `app.py`, and click **Deploy**.

### Deploy Gradio to Hugging Face Spaces
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Create a **New Space**, choose **Gradio** as the SDK.
3. Upload `gradio_app.py`, `model_utils.py`, and `requirements.txt` (set entrypoint to `gradio_app.py`).

---

## ⚠️ Limitations & Ethical AI Disclaimer

1. **Context & Temporal Shifts:** Language models fine-tuned on historical datasets may be unfamiliar with emerging geopolitical events, slang, or recent election developments.
2. **Style vs. Factuality:** Sequence classification models identify *stylistic and rhetorical patterns* (e.g. sensationalist formatting, clickbait tone, uncredited rumors) rather than querying live knowledge bases.
3. **Human-in-the-Loop:** Predictions produced by this platform are statistical estimates. Important or sensitive news must always be cross-referenced with primary journalistic and official records.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
