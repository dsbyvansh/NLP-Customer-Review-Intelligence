# NLP-Based Customer Support Intelligence System (In Progress)

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
├── notebooks/          # One notebook per phase — exploration and analysis
├── data/
│   ├── chromadb/
│   ├── cleaned/        # Preprocessed dataset (cleaned_review.csv, embeddings)
│   ├── raw/            # Raw filtered dataset (raw_reviews.csv)
|   ├── plots/          # Visualizations (UMAP projection)
├── src/                # Reusable functions imported by app and notebooks
├── app/                # Streamlit app (Phase 8)
├── Model_Card.md       # Model Card (Phase 6)
└── requirements.txt
```
> Note: ChromaDB index is not tracked in git. Run `day8_chromadb.ipynb` 
> to regenerate it locally before running the app.

---

## Progress

- [x] **Phase 1** — Problem framing, data pipeline, preprocessing (Day 1-2)
- [x] **Phase 2** — TF-IDF baseline, cosine similarity search (Day 4)
- [x] **Phase 3** — Sentence embeddings, semantic search, UMAP visualization (Day 5-6)
- [x] **Phase 4** — BERTopic topic modeling, topic naming, label assignment (Day 7)
- [x] **Phase 5** — ChromaDB vector store, retrieval system with metadata filtering (Day 8-9)
- [x] **Phase 6** — Explainability (LIME), model card, topic label audit (Day 10-12)
- [ ] **Phase 7** — RAG pipeline, Claude API integration, Confidence Gate
- [ ] **Phase 8** — Streamlit deployment

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
- **Confidence gate:** Low-confidence predictions flagged for manual review instead of auto-routing (Phase 7)
- **Streaming load:** 9.34GB JSONL file streamed line-by-line — no full download required
- **Phase ordering:** Topic modeling before vector storage — topic labels stored as ChromaDB metadata for filtered retrieval

---

## Key Findings So Far

- **TF-IDF vs Semantic Search (Phase 2-3):** TF-IDF performs well on rare, distinctive vocabulary ("incompatible", "uncomfortable") but fails on generic complaint terms ("broken", "not working") due to lack of semantic understanding. Semantic search won 9/10 queries on the same evaluation set. Hybrid retrieval identified as the production-grade solution.

- **HTML artifacts (Phase 2):** HTML entities and tags survived initial preprocessing and were only caught through vocabulary inspection post-vectorization — a reminder to always audit your feature space, not just your raw text.

- **UMAP visualization (Phase 3):** Complaint embeddings form natural clusters without labels. An isolated counterfeit/fraud cluster was discovered far from the main complaint blob — semantically distinct enough to warrant separate routing in production.

- **Topic modeling (Phase 4):** BERTopic discovered 40 actionable complaint topics from 30,157 reviews using pre-computed embeddings (58 seconds). Largest topic: Charging Issues (2,746 reviews). 39% noise rate accepted — ambiguous short reviews excluded from routing but retained for semantic retrieval. Several topics initially flagged as redundant pairs during manual review; later explainability work (Phase 6) showed two of these were not redundancy but outright mislabeling.

- **Explainability surfaces data bugs, not just predictions (Phase 6):** Built a LIME wrapper over a centroid-based topic classifier and ran it on 10 sample complaints. The heatmap exposed an anomalous word-importance signal that led to discovering two separate topic mislabeling bugs from BERTopic's `reduce_topics()` step — one topic labeled "Stylus/Pen issues" actually contained camera lens/protector complaints, and an adjacent "Camera/Camera lens issues" topic actually contained generic wear/durability complaints (plus one cross-category data leak from a non-stick pan review). Both relabeled and re-propagated through the dataframe and ChromaDB. The practical lesson: explainability tooling is as valuable for catching upstream data quality issues as it is for justifying individual predictions — no exhaustive audit of the remaining 37 topics has been performed, so similar issues may still exist.