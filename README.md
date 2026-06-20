# NLP-Based Customer Support Intelligence System (In working currently)

An end-to-end NLP pipeline that extracts complaint intelligence from Amazon reviews — assigning topic labels to incoming complaints and surfacing similar past resolved cases for faster agent resolution.

Built as a resume-grade portfolio project simulating a real support ticket intelligence system.

---

## Problem Statement

Support teams receive thousands of complaints daily. Manual triage is slow, inconsistent, and creates resolution bottlenecks. This system automates two core tasks:

1. **Topic Classification** — assigns each complaint a topic label + confidence score for routing and prioritization
2. **Similar Case Retrieval** — given a new complaint, returns the 3 most similar past resolved cases so agents resolve faster without starting from scratch

> Real support ticket data is proprietary. 1 and 2-star Amazon Cell Phone & Accessories reviews are used as a proxy — low-rated reviews are functionally complaints and closely mirror support ticket language.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | HuggingFace Datasets, Pandas, Requests |
| NLP Preprocessing | NLTK, Regex, Emoji |
| Classical Baseline | Scikit-learn (TF-IDF) |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Storage | ChromaDB |
| Topic Modeling | BERTopic, HDBSCAN, UMAP |
| Explainability | SHAP, LIME |
| RAG Pipeline | Claude API (claude-haiku-4-5), ChromaDB |
| Deployment | Streamlit, HuggingFace Spaces |

---

## Project Structure

```
nlp_ticket_intelligence/
│
├── notebooks/          # One notebook per week — exploration and analysis
├── data/
│   └── cleaned/        # Preprocessed dataset (cleaned_review.csv)
|   └── raw/            # Raw dataset (raw_reviews.csv)
├── src/                # Reusable functions imported by app and notebooks
├── app/                # Streamlit app (Week 8)
└── requirements.txt
```

---

## Progress

- [x] **Phase 1** — Problem framing, data pipeline, preprocessing (Day 1)
- [x] **Phase 2** — TF-IDF baseline, cosine similarity search (Day 4)
- [ ] **Phase 3** — Sentence embeddings, semantic search, UMAP
- [ ] **Phase 4** — ChromaDB vector store, retrieval system
- [ ] **Phase 5** — BERTopic topic modeling
- [ ] **Phase 6** — Explainability (LIME/SHAP)
- [ ] **Phase 7** — RAG pipeline
- [ ] **Phase 8** — Streamlit deployment

---

## Dataset

- **Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Category:** Cell Phones & Accessories
- **Size:** 30,157 complaints (filtered from 200K rows)
- **Filter:** 1 and 2-star verified purchase reviews only

---

## Key Design Decisions

- **Proxy data:** Low-star Amazon reviews used instead of proprietary support tickets
- **Dual preprocessing:** `text_clean` (stopwords removed) for TF-IDF; `text_clean_full` (stopwords kept) for sentence transformers
- **Confidence gate:** Low-confidence predictions flagged for manual review instead of auto-routing (Week 7)
- **Streaming load:** 9.34GB JSONL file streamed line-by-line — no full download required

## Key Findings So Far

- TF-IDF + cosine similarity performs well on queries with distinctive vocabulary
  (e.g. "incompatible", "uncomfortable") but fails on generic complaint terms
  (e.g. "broken", "not working") due to lack of semantic understanding —
  motivating the move to sentence embeddings in Phase 3.
- HTML artifacts (entities, tags) survived initial preprocessing and were only
  caught through vocabulary inspection post-vectorization — a reminder to always
  audit your feature space, not just your raw text.