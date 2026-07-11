# Model Card: NLP Ticket Intelligence System

## Model Overview
A two-stage NLP pipeline for customer complaint routing and response generation,
built without training dedicated classifiers. Stage 1 classifies incoming complaints
into one of 39 topics using a nearest-centroid approach over sentence embeddings.
Stage 2 retrieves similar past complaints from a vector store and generates a
suggested customer-facing response via a Retrieval-Augmented Generation (RAG)
pipeline with a confidence gate.

## Intended Use
- Routing/prioritizing incoming customer support complaints by topic
- Retrieving similar past complaints for agent reference (via ChromaDB)
- Generating suggested agent responses using RAG
- Providing local explanations (LIME) for why a complaint was routed to a
  given topic, to support human review and debugging

## Not Intended For
- High-stakes or compliance-sensitive classification without human review
- Complaints under ~20 words (LIME explanations become unreliable on short text;
  embedding quality degrades, affecting retrieval accuracy)
- Reviews falling into the "Noise" cluster (~37% of corpus) — these have no
  centroid and are excluded from topic-based routing entirely
- Complaints classified into "Irrelevant" topics (5 of 39 topic IDs) — these
  are excluded from RAG retrieval; the pipeline falls back to direct generation
  using topic label only
- Production use without first validating topic label accuracy against a
  larger manual sample than was done here (see Known Issues)

## Architecture

### Stage 1 — Topic Classification
| Component | Detail |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers, 384-dim) |
| Topic discovery | BERTopic, reduced from 79 → 40 topics via `reduce_topics()` |
| Classification | Cosine similarity to 39 topic centroids (1 noise topic excluded) + softmax |
| Vector store | ChromaDB, persistent, cosine HNSW index |
| Explainability | LIME (`LimeTextExplainer`), 500 perturbations per explanation |

### Stage 2 — RAG Pipeline
| Component | Detail |
|---|---|
| Retrieval | ChromaDB `retrieve_filtered()` — topic-scoped search (not full corpus) |
| Confidence gate | Top-1 cosine distance only; threshold = 0.4 |
| RAG path | Distance ≤ 0.4 → Top-3 similar complaints passed as context to LLM |
| Direct path | Distance > 0.4 or empty retrieval → topic label only passed to LLM |
| LLM | Groq API, `llama-3.1-8b-instant` (free tier) |
| Output format | Customer-facing support response |

### Confidence Gate Design Decisions
- **Top-1 distance only (not average of Top-3):** The first retrieved result
  is the most semantically similar complaint in the topic. If that result is
  already a poor match (distance > 0.4), averaging with two even worse matches
  would mask the signal. Top-1 is simpler and more interpretable.
- **Threshold = 0.4:** First observed during Phase 5 ChromaDB exploration,
  where distances above 0.4 corresponded to visually dissimilar complaints.
  Validated experimentally during Phase 7 manual evaluation on 20 complaints
  across low, boundary, high, and edge-case distance buckets.
- **Topic-scoped retrieval:** `retrieve_filtered()` constrains ChromaDB search
  to the predicted topic only. Retrieving across the full corpus risks pulling
  in semantically similar but topically unrelated complaints, which would
  mislead the LLM response.

## Training Data
Amazon Reviews 2023 — Cell Phones & Accessories category, filtered to 1–2
star reviews (30,157 complaints) as a proxy for customer support tickets.

**Proxy dataset justification:** Proprietary support ticket data is unavailable
for portfolio use. Amazon 1–2 star reviews share the core properties of support
tickets: they describe specific product failures, express customer frustration,
and are grounded in concrete product interactions. The Cell Phones & Accessories
category was chosen for topic diversity (charging, screen protection, cases,
connectivity, etc.) and complaint volume.

## Evaluation

### Topic Classification
- TF-IDF vs. semantic search head-to-head on 10 hand-written queries:
  semantic search won 9/10
- LIME explanations run on 10 sampled reviews (≥20 words, non-noise topics)
  — qualitatively confirmed strong topic separation in 9/10 cases

### RAG Pipeline
Manual evaluation on 20 complaints, stratified by Top-1 distance bucket:
- 8 boundary cases (distance 0.2–0.4, weighted toward threshold)
- 4 low-distance cases (distance < 0.2, clear RAG path)
- 4 high-distance cases (distance > 0.4, clear direct generation path)
- 4 edge cases (short complaints under 20 words)

Responses scored on a 1–3 scale (1=poor, 2=acceptable, 3=good).

**Finding 1 — RAG improves response quality:**
RAG path averaged 2.81/3 vs. direct generation at 2.25/3 across 20 evaluated
complaints. Retrieval consistently produced more specific, actionable responses.

**Finding 2 — Topic misclassification is frequent but LLM partially compensates:**
Topic labels were misclassified in approximately 10 of 20 evaluated complaints.
In RAG cases, retrieved similar complaints provided enough grounding for the LLM
to respond correctly to the actual complaint text despite the wrong topic label.
In direct generation cases, misclassification combined with no retrieved context
produced the worst responses in the evaluation (e.g. row 14, score 1 — complaint
about sewing quality misclassified as "Glass Screen related issues", opening line
referenced wrong product).

**Finding 3 — Threshold 0.4 is valid:**
The confidence gate correctly separated high-confidence retrievals from weak ones.
Direct generation failures were caused by short/thin complaint text (< 5 words),
not by the threshold being too loose or too strict. The threshold itself held up
across all boundary cases (distance 0.35–0.40), all of which produced quality
responses.

**Finding 4 — Hallucination pattern identified in RAG path:**
In 2 of 16 RAG cases, the LLM hallucinated a prior customer relationship
("Based on your previous interactions", "We've noticed a pattern in your reports").
This appears triggered when retrieved similar complaints describe recurring issues —
the LLM incorrectly infers a history with the current customer. Fix: explicitly
instruct the LLM in the system prompt never to reference prior customer history.

**Finding 5 — Short complaint failure mode:**
Complaints under 5 words produced poor responses on both paths (avg score 2.0).
Direct generation has no retrieved context to compensate for thin input. Proposed
fix: add a length gate upstream — if complaint is under 5 words, prompt the user
to describe their issue in more detail before routing to the pipeline.

## Known Issues / Limitations
1. **Noise rate:** ~37% of the corpus falls outside the 39 defined topics
   and is excluded from routing
2. **Irrelevant topics:** 5 of 39 topic IDs carry the label "Irrelevant" —
   BERTopic assigned them a cluster ID but manual audit found no coherent
   complaint category. These are excluded from RAG retrieval and evaluation.
   Complaints classified into these topics receive direct generation only.
3. **Topic mislabeling (found and fixed, recurring pattern):** Three
   separate label corrections were made after Phase 6 explainability work
   began, all stemming from `reduce_topics()` merge artifacts:
   - Topic originally "Stylus/Pen issues" → actually camera lens/protector
     complaints → relabeled "Camera lens/protector issues"
   - Topic originally "Camera/Camera lens issues" → actually generic
     wear/durability complaints (scratching, peeling) across unrelated
     product types → relabeled "Generic wear/durability complaints"
   - These were caught via LIME explainability (case 1) and manual audit
     of adjacent topics (case 2). No exhaustive audit of the remaining 37
     topics has been performed; similar mislabeling may exist elsewhere.
     This recurring pattern suggests `reduce_topics()` output requires
     systematic manual verification, not spot-checking, before trusting
     auto-generated or inherited labels
4. **Small-sample explainability:** LIME was run on only 10 reviews; this is
   illustrative, not a statistically representative evaluation of
   explanation quality
5. **LIME explains the predicted class only:** `as_list()` does not show how
   words influenced probabilities for other topics
6. **Centroid quality varies by topic size:** Topics with few member reviews
   (e.g. under 100) produce noisier centroids than large topics like
   Charging Issues (2,746 reviews)
7. **Minor cross-category contamination:** At least one review unrelated
   to Cell Phones & Accessories (a non-stick cooking pan review) was found
   within the corpus during the topic 17 investigation, likely from the
   original Phase 1 streaming/filtering step. Not exhaustively scanned or
   removed; affects an estimated <0.01% of the corpus based on spot-check
8. **RAG threshold not production-validated:** The 0.4 confidence gate
   threshold was selected from observation and validated on 20 manually
   evaluated complaints. It has not been validated against a held-out set
   with ground-truth labels or user feedback signals.

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
- RAG pipeline added (Phase 7): Groq API, confidence gate at distance 0.4,
  topic-scoped retrieval via `retrieve_filtered()`
- Phase 7 evaluation completed: 20-complaint manual eval surfaced hallucination
  pattern, short-complaint failure mode, and frequent topic misclassification;
  threshold 0.4 validated; findings documented in evaluation section above