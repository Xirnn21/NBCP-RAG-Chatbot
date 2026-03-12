# NBCP RAG Chatbot — Gemma 2B IT vs. Qwen2.5-1.5B-Instruct

> **Development and Evaluation of a Domain-Specific RAG Chatbot for the National Building Code of the Philippines**
> 
> *Patricia Mae G. Dela Cruz & Nathaniel H. Dumayas — Mapua University, School of Information Technology*

---

## Overview

This project builds and evaluates a **Retrieval-Augmented Generation (RAG) chatbot** for question answering over the [National Building Code of the Philippines (NBCP)](https://ldr.senate.gov.ph/legislative%2Bissuances/Presidential%20Decree%20No.%201096%2C%20s.%201977) (Presidential Decree No. 1096). The chatbot allows users to query complex regulatory provisions in natural language, grounding all answers strictly in the source document.

Two lightweight, open-weight instruction-tuned models are compared as generation backends under **identical retrieval conditions**:
- **Gemma 2B IT** (`google/gemma-2b-it`)
- **Qwen2.5-1.5B-Instruct** (`Qwen/Qwen2.5-1.5B-Instruct`)

---

## 🚀 Live Demo

**Deployed on Hugging Face Spaces:** [https://huggingface.co/spaces/dcpati/Ai3-Project](https://huggingface.co/spaces/dcpati/Ai3-Project)

---

## Key Results

| Metric | Gemma 2B IT | Qwen2.5-1.5B |
|---|---|---|
| Recall@k | **1.0** | **1.0** |
| Precision@k | 0.2 | 0.2 |
| Token F1 | **0.6078** | 0.5637 |
| Token Recall | **0.5963** | 0.5354 |
| Token Precision | 0.6405 | **0.6786** |
| Numeric Recall | **0.9048** | 0.7143 |
| Unit Recall | **0.9048** | 0.7619 |
| Generation Failures (F1 < 0.30) | **2 / 21** | 3 / 21 |

Both models achieved perfect retrieval recall. Gemma outperformed Qwen on overall answer completeness and numeric/unit precision, making it the more suitable model for compliance-oriented NBCP question answering.

---

## System Architecture

```
NBCP PDF
   │
   ▼
PDF Parsing (pypdf)
   │
   ▼
Text Chunking (900 chars, 150 overlap)
   │
   ▼
Dense Embeddings (all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Index (IndexFlatIP)
   │
   ▼
Top-K Retrieval (k=20) ──► Constrained Prompt
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                    Gemma 2B IT       Qwen2.5-1.5B-Instruct
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                              Answer + Sources
```

### Components

- **Document Parsing** — `pypdf` extracts page-level text from the NBCP PDF
- **Chunking** — 900-character overlapping chunks (150-char overlap) to preserve clause continuity
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` for dense semantic encoding
- **Vector Store** — FAISS `IndexFlatIP` for efficient inner-product similarity search
- **Generation** — Prompt-constrained LLM inference with strict numeric/unit preservation rules and a "Not specified in the document" fallback
- **Deployment** — Gradio web interface hosted on Hugging Face Spaces

---

## Evaluation

The chatbot was evaluated on **21 held-out, user-style questions** spanning multiple NBCP chapters (General Provisions, Administration, Permits & Inspection, Light & Ventilation, Building Projections, Pedestrian Safety). Metrics include:

- **Token F1 / Precision / Recall** — overlap-based answer quality
- **Numeric Recall** — fraction of gold numeric values preserved in the answer
- **Unit Recall** — fraction of gold measurement units preserved
- **Recall@k / Precision@k** — retrieval quality at top-k=5

---

## Repository Structure

```
├── AI3_NBCP__GEMMA___QWEN_.ipynb   # Main notebook (pipeline + evaluation)
├── AI3_Machine_Project.docx        # Full project report / paper
└── README.md
```

---

## Installation

```bash
pip install pypdf sentence-transformers faiss-cpu transformers \
            accelerate huggingface_hub rouge-score pandas gradio
```

> ℹ️ Designed to run on **Google Colab** (GPU recommended). The notebook will prompt you to upload the NBCP PDF at runtime.

---

## Usage

1. Open `AI3_NBCP__GEMMA___QWEN_.ipynb` in Google Colab.
2. Run the setup cells to install dependencies and authenticate with Hugging Face (required for Gemma).
3. Upload your copy of the **NBCP PDF** when prompted.
4. Run through the sections to build the index, load models, and launch the Gradio chatbot.
5. Use the evaluation cells (Section 5–7) to reproduce the reported metrics.

### Example Query

```python
ask_chatbot(
    "Under Section 103(a), what buildings/structures does the Code apply to?",
    model_name="gemma"  # or "qwen"
)
```

---

## Models

| Model | Source | Parameters | Notes |
|---|---|---|---|
| Gemma 2B IT | `google/gemma-2b-it` | ~2B | Requires HF login & license acceptance |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.5B | Publicly available |

---

## Authors

| Name | Email |
|---|---|
| Patricia Mae G. Dela Cruz | pmgdelacruz@mymail.mapua.edu.ph |
| Nathaniel H. Dumayas | nhdumayas@mymail.mapua.edu.ph |

*Mapua University — School of Information Technology, Makati, Philippines*

---

## References

1. Senate of the Philippines, *Presidential Decree No. 1096, s. 1977* (NBCP)
2. P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
3. N. Reimers & I. Gurevych, *Sentence-BERT*, EMNLP 2019
4. J. Johnson et al., *Billion-scale similarity search with GPUs* (FAISS)
5. Google DeepMind, *Gemma: Open Models Based on Gemini Technology*, 2024
6. Alibaba Cloud, *Qwen2.5 Technical Report*, 2024
