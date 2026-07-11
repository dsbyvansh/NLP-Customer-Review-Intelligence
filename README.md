# NLP-Based Customer Support Intelligence System

An end-to-end NLP pipeline that extracts complaint intelligence from Amazon reviews — assigning topic labels to incoming complaints, retrieving similar past cases, and generating suggested customer-facing responses via a RAG pipeline.

---

## Live Demo
🚀 [Try the app here](https://nlp-customer-review-intelligence-arftozvxbssveh7jpkubfx.streamlit.app/)

---

## Problem Statement

Support teams receive thousands of complaints daily. Manual triage is slow, inconsistent, and creates resolution bottlenecks. This system automates three core tasks:

1. **Topic Classification** — assigns each complaint a topic label + confidence score for routing and prioritization
2. **Similar Case Retrieval** — given a new complaint, returns the 3 most similar past cases so agents resolve faster without starting from scratch
3. **Response Generation** — generates a suggested customer-facing reply using a RAG pipeline with a confidence gate

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
| Explainability | LIME |
| RAG Pipeline | Groq API (llama-3.1-8b-instant), ChromaDB |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
nlp_ticket_intelligence/
│
├── notebooks/          # One notebook per phase — exploration and analysis
├── data/
│   ├── chromadb/       # ChromaDB vector store (tracked via Git LFS)
│   ├── cleaned/        # Preprocessed dataset (cleaned_reviews.csv, embeddings)
│   ├── raw/            # Raw filtered dataset (raw_reviews.csv)
│   └── plots/          # Visualizations (UMAP projection)
├── src/                # Reusable modules imported by app.py
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── retrieval.py
│   └── pipeline.py
├── app.py              # Streamlit app entry point
├── pages
    └──result.py        # Result page for streamlit app
├── MODEL_CARD.md       # Model card (updated through Phase 7)
└── requirements.txt
```

---

## Progress

- [x] **Phase 1** — Problem framing, data pipeline, preprocessing (Day 1-2)
- [x] **Phase 2** — TF-IDF baseline, cosine similarity search (Day 4)
- [x] **Phase 3** — Sentence embeddings, semantic search, UMAP visualization (Day 5-6)
- [x] **Phase 4** — BERTopic topic modeling, topic naming, label assignment (Day 7)
- [x] **Phase 5** — ChromaDB vector store, retrieval system with metadata filtering (Day 8-9)
- [x] **Phase 6** — Explainability (LIME), model card, topic label audit (Day 10-12)
- [x] **Phase 7** — RAG pipeline, Groq API, confidence gate, 20-complaint evaluation (Day 13-16)
- [x] **Phase 8** — Streamlit app, HuggingFace Spaces deployment (Day 17+)

> Note: Topic modeling (Phase 4) precedes vector storage (Phase 5) so that topic labels
> can be stored as metadata in ChromaDB for filtered retrieval.

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
- **Confidence gate:** Top-1 cosine distance threshold of 0.4 — below triggers RAG, above triggers direct generation with topic label only
- **Length gate:** Complaints under 5 words return `None` — UI prompts user to describe issue further
- **Topic-scoped retrieval:** ChromaDB search constrained to predicted topic only, preventing cross-topic context contamination
- **Streaming load:** 9.34GB JSONL file streamed line-by-line — no full download required
- **Phase ordering:** Topic modeling before vector storage — topic labels stored as ChromaDB metadata for filtered retrieval
- **src/ module structure:** All reusable functions refactored into explicit-parameter modules — no globals, clean imports

---

## Key Findings

- **TF-IDF vs Semantic Search (Phase 2-3):** Semantic search won 9/10 queries on the same evaluation set. TF-IDF performs well on rare distinctive vocabulary but fails on generic complaint terms ("broken", "not working") due to lack of semantic understanding.

- **HTML artifacts (Phase 2):** HTML entities and tags survived initial preprocessing — only caught through vocabulary inspection post-vectorization. Always audit your feature space, not just raw text.

- **UMAP visualization (Phase 3):** Complaint embeddings form natural clusters without labels. An isolated counterfeit/fraud cluster was discovered far from the main blob — semantically distinct enough to warrant separate routing in production.

- **Topic modeling (Phase 4):** BERTopic discovered 40 actionable complaint topics from 30,157 reviews. Largest topic: Charging Issues (2,746 reviews). 39% noise rate accepted — ambiguous short reviews excluded from routing but retained for semantic retrieval.

- **Explainability surfaces data bugs, not just predictions (Phase 6):** LIME heatmap exposed anomalous word-importance signals leading to two topic mislabeling bugs from BERTopic's `reduce_topics()` step — "Stylus/Pen issues" relabeled to "Camera lens/protector issues" and "Camera/Camera lens issues" relabeled to "Generic wear/durability complaints." One cross-category data leak (cooking pan review) also discovered. Lesson: explainability tooling is as valuable for catching upstream data quality issues as for justifying individual predictions.

- **RAG evaluation (Phase 7):** RAG averaged 2.81/3 vs direct generation at 2.25/3 across 20 manually scored complaints. Topic misclassification observed in ~10/20 cases — RAG compensated via retrieved context while direct generation failed when misclassification combined with thin input. Hallucination pattern identified: LLM invented prior customer history when retrieved context described recurring issues. Fixed via system prompt. Threshold 0.4 validated.