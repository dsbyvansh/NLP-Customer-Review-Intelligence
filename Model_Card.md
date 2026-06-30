# Model Card: Complaint Topic Classifier (Centroid-Based)

## Model Overview
A lightweight topic classification system for customer complaint routing,
built without training a dedicated classifier. Instead, it uses a
**nearest-centroid approach** over sentence embeddings: each of 39 topics
is represented by the mean embedding of its member complaints, and new
text is classified by cosine similarity to these centroids.

## Intended Use
- Routing/prioritizing incoming customer support complaints by topic
- Retrieving similar past complaints for agent reference (via ChromaDB)
- Providing local explanations (LIME) for why a complaint was routed to a
  given topic, to support human review and debugging

## Not Intended For
- High-stakes or compliance-sensitive classification without human review
- Reviews under ~20 words (LIME explanations become unreliable on short text)
- Reviews falling into the "Noise" cluster (~37% of corpus) — these have no
  centroid and are excluded from topic-based routing entirely
- Production use without first validating topic label accuracy against a
  larger manual sample than was done here (see Known Issues)

## Architecture
| Component | Detail |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers, 384-dim) |
| Topic discovery | BERTopic, reduced from 79 → 40 topics via `reduce_topics()` |
| Classification | Cosine similarity to 39 topic centroids (1 noise topic excluded) + softmax |
| Vector store | ChromaDB, persistent, cosine HNSW index |
| Explainability | LIME (`LimeTextExplainer`), 500 perturbations per explanation |

## Training Data
Amazon Reviews 2023 — Cell Phones & Accessories category, filtered to 1–2
star reviews (30,157 complaints) as a proxy for customer support tickets.

## Evaluation
- TF-IDF vs. semantic search head-to-head on 10 hand-written queries:
  semantic search won 9/10
- LIME explanations run on 10 sampled reviews (≥20 words, non-noise topics)
  — qualitatively confirmed strong topic separation in 9/10 cases

## Known Issues / Limitations
1. **Noise rate:** ~37% of the corpus falls outside the 39 defined topics
   and is excluded from routing
2. **Topic mislabeling (found and fixed, recurring pattern):** Three
   separate label corrections were made after Phase 6 explainability work
   began, all stemming from `reduce_topics()` merge artifacts:
   - Topic originally "Stylus/Pen issues" → actually camera lens/protector
     complaints → relabeled "Camera lens/protector issues"
   - Topic originally "Camera/Camera lens issues" → actually generic
     wear/durability complaints (scratching, peeling) across unrelated
     product types → relabeled "Generic wear/durability complaints"
   - These were caught via LIME explainability (case 1) and manual audit
     of adjacent topics (case 2). No exhaustive audit of the remaining 37
     topics has been performed; similar mislabeling may exist elsewhere
     and has not been ruled out. This recurring pattern suggests
     `reduce_topics()` output requires systematic manual verification,
     not spot-checking, before trusting auto-generated or inherited labels
3. **Small-sample explainability:** LIME was run on only 10 reviews; this is
   illustrative, not a statistically representative evaluation of
   explanation quality
4. **LIME explains the predicted class only:** `as_list()` does not show how
   words influenced probabilities for other topics
5. **Centroid quality varies by topic size:** Topics with few member reviews
   (e.g. under 100) produce noisier centroids than large topics like
   Charging Issues (2,746 reviews)
6. **Minor cross-category contamination:** At least one review unrelated
   to Cell Phones & Accessories (a non-stick cooking pan review) was found
   within the corpus during the topic 17 investigation, likely from the
   original Phase 1 streaming/filtering step. Not exhaustively scanned or
   removed; affects an estimated <0.01% of the corpus based on spot-check

## Update History
- Initial topic assignment used pre-reduction (79-topic) labels — bug found
  when centroid count (78) didn't match expected reduced topic count (40);
  fixed by correctly applying `reduce_topics()` output to the dataframe
- "Stylus/Pen issues" relabeled to "Camera lens/protector issues" after
  LIME explainability surfaced the mislabel (see `chroma_db.ipynb` and
  `topic_modelling.ipynb` for full fix documentation)
- "Camera/Camera lens issues" relabeled to "Generic wear/durability
  complaints" after auditing the topic adjacent to the first fix; also
  surfaced minor cross-category data contamination (see Known Issues)