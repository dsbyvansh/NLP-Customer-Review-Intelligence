# Customer Review Support Intelligence (In working)

- Amazon receives thousands of customer complaints daily across product categories.
Identifying, routing, and resolving these complaints manually is slow and inconsistent.
- Support agents read each complaint individually to understand the issue and find a
resolution — a process that creates bottlenecks and inconsistent response quality.
- We built a two-capability NLP pipeline using 1 and 2-star Amazon reviews as a proxy
for support tickets (low-rated reviews are functionally complaints and closely mirror
support ticket language and intent).

**What The System Does:**
1. Each complaint is assigned a topic label with a confidence score — so leads can
   route and prioritize without reading the full text
2. Given any new complaint, the system retrieves the 3 most similar past resolved
   complaints — so agents have an immediate reference for resolution

**How We Measure Success:**
- Reduction in average manual triage time (time taken to review)
- Relevance of top-3 retrieved complaints (measured by human evaluation)
- Topic coherence (are the discovered topics meaningful and distinct)