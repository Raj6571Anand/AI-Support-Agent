# Amazon Help AI Support Agent

> Hiver SDE Intern Take-Home Assignment  
> An AI customer support agent for **AmazonHelp** that classifies intents, drafts grounded replies, and makes escalation decisions — with rigorous evaluation proving it works.

## Quick Start — Reproduce Results in Under 10 Minutes

```bash
# 1. Clone and install
git clone <repo-url>
cd Hiver
pip install -r requirements.txt

# 2. Set your Groq API key (free: https://console.groq.com)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Download the dataset
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data --unzip

# 4. Run the full pipeline
python run_all.py

# 5. Launch the interactive demo
python src/demo/app.py
```

**Pre-computed results are included** — you can skip steps 3-4 and go straight to the demo or read the evaluation results in `results/`.

## Architecture

```
Customer Tweet
      │
      ▼
┌──────────────────────┐
│ 1. Preprocessing     │  Filter English, clean @mentions, reconstruct threads
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. Intent Classifier │  BGE embeddings + cosine similarity to intent exemplars
│    + Confidence Score│  (LLM fallback for low-confidence cases)
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌────────────────────┐
│ 3. Esc. │  │ 4. Reply Generator │
│ Decision│  │   Intent-specific  │
│ (rules +│  │   RAG + LLM        │
│  data)  │  └────────────────────┘
└─────────┘
```

## Brand: AmazonHelp

| Metric | Value |
|--------|-------|
| Total brand replies in dataset | 169,840 |
| Customer messages replied to | 154,985 |
| Unique customers | 71,049 |
| Processed subsample | 5,000 conversations |
| Language | English only (75% of brand data) |

## Intent Taxonomy (8 Intents)

| Intent | Description | % of Data |
|--------|-------------|-----------|
| `order_issue` | Delivery delays, missing packages, tracking | 30.8% |
| `refund_return` | Refund requests, return process, overcharges | 6.3% |
| `account_access` | Login problems, locked accounts, verification | 3.8% |
| `subscription` | Prime membership, billing, cancellation | 8.9% |
| `product_issue` | Defective products, device issues (Kindle, Alexa) | 5.5% |
| `payment_billing` | Payment failures, gift cards, charge disputes | 2.8% |
| `feedback` | Praise/complaints with no actionable request | 9.0% |
| `general_inquiry` | How-to questions, policy questions | catch-all |

## Evaluation Results

See [REPORT.md](REPORT.md) for full analysis.

## Project Structure

```
Hiver/
├── run_all.py                    # Master pipeline (reproduce all results)
├── requirements.txt
├── .env.example
├── REPORT.md                     # Full evaluation report (6 pages)
├── DECISIONS.md                  # 15 non-obvious decisions with reasoning
├── src/
│   ├── config.py                 # Central configuration
│   ├── pipeline/
│   │   ├── preprocess.py         # Data cleaning + thread reconstruction
│   │   ├── build_vector_store.py # Embedding + ChromaDB indexing
│   │   ├── classifier.py         # Intent classification (embedding + LLM)
│   │   ├── reply_generator.py    # RAG-based reply drafting
│   │   └── escalation.py         # Escalation decision engine
│   ├── evaluation/
│   │   ├── golden_set_builder.py # Stratified sampling + labeling
│   │   ├── eval_classification.py# F1, confusion matrix, bootstrap CIs
│   │   ├── eval_replies.py       # LLM-as-judge with inter-rater agreement
│   │   └── eval_escalation.py    # Precision/recall on escalation
│   └── demo/
│       └── app.py                # Gradio interactive demo
├── data/
│   └── amazon_processed.json     # 5,000 processed conversation pairs
├── golden_set/
│   └── golden_set_labelled.json  # 200 hand-labelled evaluation examples
├── results/
│   ├── classification_results.json
│   ├── reply_eval_results.json
│   └── escalation_results.json
└── chroma_db/                    # Vector store (auto-generated)
```

## Tech Stack (100% Free)

| Component | Tool |
|-----------|------|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) |
| LLM | Groq free tier (`qwen/qwen3.8-27b`) |
| Vector Store | ChromaDB (local) |
| Demo | Gradio |
| Evaluation | scikit-learn + custom LLM judge |

## License

Dataset: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Kaggle)

## Citations

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by thoughtvector
- Embedding model: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- LLM: [Qwen 3.8 27B](https://huggingface.co/Qwen) via Groq
- AI coding assistant used during development (all code reviewed and understood by author)
