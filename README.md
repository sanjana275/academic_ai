# 🔬 LLM X-Ray: Interactive Visualization of LLM Inference

An interactive, local mechanistic interpretability web application that allows you to look inside an autoregressive Large Language Model (LLM) during inference.

Built for **`Qwen/Qwen2.5-1.5B-Instruct`** (with automatic support for `Qwen/Qwen2.5-0.5B-Instruct` for quick CPU testing), running completely locally on your personal laptop without any external API calls.

---

## 🌟 Core Inference Pipeline

```
Input Prompt ➔ Tokenizer ➔ Token IDs ➔ Input Embeddings ➔ 28 Transformer Blocks ➔ Self-Attention ➔ Hidden States ➔ LM Head ➔ Logits ➔ Softmax ➔ Next Token ➔ Response
```

---

## 📁 Project Architecture

```
LLM-XRay/
├── app.py              # Streamlit Web Application with interactive tabs & controls
├── model.py            # Hugging Face model loading, forward pass & autoregressive tracer
├── tokenizer.py        # Tokenizer extraction, decoding & stylized token chip formatting
├── visualization.py    # Plotly interactive visualizations (PCA, heatmaps, bar charts)
└── requirements.txt    # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run from the project folder:
```bash
streamlit run app.py
```
Or if running from `LLM-XRay`:
```bash
streamlit run LLM-XRay/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧩 Features & Inspection Capabilities

| Step | Feature | Description |
|---|---|---|
| **Step 3** | **Interactive Chat** | Enter custom text or select presets (*"What is AI?"*, *"The capital of France is"*, etc.). |
| **Step 4** | **Tokenization** | Visual token chips, token IDs, lengths, and BPE breakdown. |
| **Step 5** | **Input Embeddings** | 1,536-dimensional vectors, L2 norm, statistics, and **2D PCA projection scatter plot**. |
| **Step 6** | **Transformer Architecture** | High-level diagram and parameter breakdown of all 28 decoder layers. |
| **Step 7** | **Multi-Head Attention** | Interactive causal attention heatmap with layer (0–27) and head selectors. |
| **Step 8** | **Hidden States Depth** | Token trajectory through the 28 layers projected with 2D PCA, and layer cosine similarity matrix. |
| **Step 9** | **Logits & Probabilities** | Top-K candidate predictions bar chart with percentages and entropy metric. |
| **Step 10** | **Step-by-Step Tracer** | Interactive token-by-token generation replay showing competing candidates at each step. |

---

## ⚙️ Interactive Controls (Sidebar)

- **Model Selection:** Default is `Qwen/Qwen2.5-1.5B-Instruct`. You can also choose `Qwen/Qwen2.5-0.5B-Instruct` for instantaneous responses on CPU, or enter a custom Hugging Face model ID.
- **Temperature:** Controls randomness (0.0 = greedy/deterministic, >1.0 = creative).
- **Max New Tokens:** Adjusts the number of tokens generated step-by-step.
- **Top-K & Top-P:** Nucleus and top-k probability truncation parameters.
- **Layer & Attention Head Selectors:** Dive into specific attention heads across all 28 layers.
