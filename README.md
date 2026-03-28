# Real-Time Recommendation Engine

![Java](https://img.shields.io/badge/Java-17%2B-blue?logo=openjdk)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2.5-brightgreen?logo=springboot)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen)
![Redis](https://img.shields.io/badge/cache-Redis-red?logo=redis)
![From Scratch](https://img.shields.io/badge/TF--IDF%20%7C%20LSH%20%7C%20Cosine-from%20scratch-orange)

A recommendation engine built from scratch in Java/Spring Boot — no recommendation library, no ML framework. Given a user's viewing history it finds the most similar items using **TF-IDF vectorization**, **cosine similarity**, and **Locality-Sensitive Hashing (LSH)** for scalable approximate nearest-neighbor search. Results are cached in Redis with configurable TTL for sub-millisecond repeated queries.

This is the same algorithmic stack that powers recommendation systems at scale: convert content to vectors, measure vector closeness, use probabilistic indexing to avoid scoring every item on every request.

---

## Features

- **TF-IDF Vectorization** — converts item descriptions into numeric vectors; vocabulary capped at top-N most discriminating terms (default 500, actual ~456 for 50 items)
- **Cosine Similarity** — measures the angle between vectors; robust to description length differences; returns 0.0 on zero vectors (no NaN)
- **Locality-Sensitive Hashing (LSH)** — random-projection hashing groups similar items into the same bucket; reduces candidate set from O(N) to O(N / 2^k); fixed seed=42 for reproducibility
- **Hamming-distance-1 expansion** — if the exact LSH bucket is empty, expands search to k adjacent buckets before falling back to full scan
- **Cache-Aside with Redis** — check Redis first → on MISS compute + store with TTL → return; evict on new view event
- **Non-fatal Redis failures** — all cache operations are try/catch wrapped; a Redis outage degrades gracefully to no-cache mode
- **Configurable everything** — top-N, TTL, hash bits, vocabulary size all in `application.properties`
- **97 tests** — unit tests for every algorithm layer, web layer tests via MockMvc, full-context integration test with mocked Redis
- **Zero recommendation libraries** — every layer built from scratch in plain Java

---

## Table of Contents

1. [What is a Recommendation Engine?](#1-what-is-a-recommendation-engine)
2. [Why This Architecture?](#2-why-this-architecture)
3. [System Architecture](#3-system-architecture)
4. [Startup Pipeline — How the system initializes](#4-startup-pipeline)
5. [Request Pipeline — How a recommendation is computed](#5-request-pipeline)
6. [Core Algorithms — Deep Dive](#6-core-algorithms--deep-dive)
   - [TF-IDF Vectorization](#61-tf-idf-vectorization)
   - [Cosine Similarity](#62-cosine-similarity)
   - [Locality-Sensitive Hashing (LSH)](#63-locality-sensitive-hashing-lsh)
   - [Cache-Aside Pattern](#64-cache-aside-pattern)
7. [Component Deep-Dive](#7-component-deep-dive)
   - [DataIngestionService](#71-dataingestionservice)
   - [TextPreprocessor](#72-textpreprocessor)
   - [TfIdfVectorizer](#73-tfidfvectorizer)
   - [RandomProjection](#74-randomprojection)
   - [LSHIndex](#75-lshindex)
   - [RecommendationEngine](#76-recommendationengine)
   - [RecommendationCacheService](#77-recommendationcacheservice)
   - [UserProfileRepository](#78-userprofilerepository)
   - [RecommendationController](#79-recommendationcontroller)
   - [ApplicationStartupRunner](#710-applicationstartuprunner)
8. [API Reference](#8-api-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Data Format](#10-data-format)
11. [Technology Stack](#11-technology-stack)
12. [Project Structure](#12-project-structure)
13. [Building and Running](#13-building-and-running)
14. [Test Coverage](#14-test-coverage)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)

---

## 1. What is a Recommendation Engine?

A recommendation engine answers one question: *"Given what this user has liked before, what else will they like?"*

There are three main approaches:

| Approach | How it works | Limitation |
|---|---|---|
| **Collaborative filtering** | Find users similar to you; recommend what they liked | Requires a large user base; cold-start problem |
| **Content-based filtering** | Find items similar to what you've already liked | Requires rich item descriptions |
| **Hybrid** | Combines both | More complex, more data needed |

This engine uses **content-based filtering**. Item descriptions are converted to numeric vectors, and similarity is measured by the angle between those vectors. A user's preferences are represented as the average of the vectors of items they've viewed. The closest items in vector space become the recommendations.

No user-to-user comparison. No matrix factorization. No model training. Just: *measure how similar the content is.*

---

## 2. Why This Architecture?

**The naive approach** — score every item against every user on every request — does not scale:

```
  50 items   ×  456 dimensions  ×  10 users/sec  =    228 000 multiplications/sec   ✓ trivial
  1M  items  ×  500 dimensions  ×  100 users/sec =  50 000 000 000 multiplications/sec   ✗ impossible
```

**LSH solves this.** Items with similar vectors hash to the same bucket. At query time only that bucket is scored — typically 3–10 items instead of all N. The engine stays fast as the catalogue grows.

**Redis solves repeated queries.** A user requesting recommendations twice within the TTL window gets the cached answer in < 1 ms instead of recomputing. This is critical when the same user opens the app repeatedly.

---

## 3. System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              REAL-TIME RECOMMENDATION ENGINE  —  FULL ARCHITECTURE             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

                              ┌─────────────────────────────┐
                              │           CLIENT            │
                              │  GET /recommend/{userId}    │
                              │  POST /users/{id}/items/{id}│
                              │  DELETE /cache/{userId}     │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                     ╔═══════════════════════════════════════════╗
                     ║       RecommendationController            ║
                     ║           REST API Layer                  ║
                     ╚═══════════════════╦═══════════════════════╝
                                         ║
              ╔══════════════════════════╩═══════════════════════════╗
              ║                CACHE-ASIDE LAYER                     ║
              ║                                                      ║
              ║   RecommendationCacheService                         ║
              ║   ─────────────────────────                          ║
              ║   get("rec:userId")  ──────────────────────────────▶ ║
              ║                      ◀─────────────────── HIT/MISS  ║
              ║              ┌─────────────────────────────────────┐ ║
              ║              │            Redis                    │ ║
              ║              │   key: "rec:{userId}"               │ ║
              ║              │   value: JSON array                 │ ║
              ║              │   TTL: configurable (default 10m)   │ ║
              ║              └─────────────────────────────────────┘ ║
              ╚════════════════╦═════════════════════════════════════╝
                               ║  (on MISS — compute fresh)
              ╔════════════════╩═════════════════════════════════════╗
              ║             RECOMMENDATION LAYER                     ║
              ║                                                      ║
              ║   UserProfileRepository                              ║
              ║   ──────────────────────                             ║
              ║   findById(userId) ──▶ [item_2, item_3, item_4]      ║
              ║                                                      ║
              ║   RecommendationEngine                               ║
              ║   ─────────────────────                              ║
              ║   1. fetch vectors for viewed items                  ║
              ║   2. average ──▶ preference vector                   ║
              ║   3. LSHIndex.getCandidates(prefVec) ─────────────▶  ║
              ║                          ◀──── candidate item list   ║
              ║   4. CosineSimilarity.compute() for each candidate   ║
              ║   5. filter viewed, sort desc, top-N                 ║
              ╚════════════════╦═════════════════════════════════════╝
                               ║
              ╔════════════════╩═════════════════════════════════════╗
              ║              VECTOR & INDEX LAYER                    ║
              ║                                                      ║
              ║   LSHIndex                  DataIngestionService     ║
              ║   ────────────              ────────────────────     ║
              ║   buckets map               ConcurrentHashMap        ║
              ║   exact bucket lookup       findById()               ║
              ║   Hamming-1 expansion       getAllItems()             ║
              ║   full fallback                                      ║
              ║                                                      ║
              ║   RandomProjection          TfIdfVectorizer          ║
              ║   ─────────────────         ───────────────          ║
              ║   k random hyperplanes      vocabulary + IDF scores  ║
              ║   fixed seed=42             fit() → transform()      ║
              ╚══════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════╗
║                         STARTUP PIPELINE (once, on boot)                       ║
║                                                                                ║
║  items.csv ──▶ DataIngestionService ──▶ TfIdfVectorizer ──▶ LSHIndex.build()  ║
║                                         fit + transform       hash all vectors ║
║                                         all items             into buckets     ║
║                                                                                ║
║  ApplicationStartupRunner seeds 5 demo UserProfiles after index is ready       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Startup Pipeline

```
ApplicationStartupRunner.run()
         │
         ▼
╔══════════════════════════════════════════════════════════════╗
║  STEP 1 — Ingest Items                                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   DataIngestionService.ingest("/data/items.csv")             ║
║                                                              ║
║   for each CSV row:                                          ║
║     Item(id, name, description)                              ║
║     → ConcurrentHashMap.put(id, item)                        ║
║                                                              ║
║   Result: 50 Item objects in memory, no vectors yet          ║
║   Log: "Ingested 50 items from /data/items.csv"              ║
╚══════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════╗
║  STEP 2 — Fit Vectorizer                                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   TfIdfVectorizer.fit(items)                                 ║
║                                                              ║
║   for each item:                                             ║
║     tokenize(description)  →  Set<String> uniqueTerms        ║
║     for each term: docFrequency[term]++                      ║
║                                                              ║
║   for each term:                                             ║
║     IDF = log( (N+1) / (df+1) )                              ║
║                                                              ║
║   sort by IDF desc → keep top-500  (actual: ~456 terms)      ║
║   → vocabulary: term → index in vector                       ║
║                                                              ║
║   Log: "TF-IDF fitted on 50 documents — vocabulary size: 456"║
╚══════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════╗
║  STEP 3 — Transform All Items                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   for each item:                                             ║
║     tokens = tokenize(description)                           ║
║     for each vocab term at index i:                          ║
║       vector[i] = TF(term, item) × IDF(term)                 ║
║     item.setVector(vector)   // double[456]                  ║
║                                                              ║
║   Log: "TF-IDF vectors computed for 50 items (dim: 456)"     ║
╚══════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════╗
║  STEP 4 — Build LSH Index                                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   LSHIndex.build(items)                                      ║
║                                                              ║
║   RandomProjection.initialize(k=4, dim=456):                 ║
║     rng = new Random(42L)   // fixed seed                    ║
║     for i in 0..3:                                           ║
║       plane[i] = 456 Gaussian random values                  ║
║       normalize to unit length                               ║
║                                                              ║
║   for each item:                                             ║
║     hash = project(item.vector)  →  e.g. "0110"             ║
║     buckets["0110"].add(item)                                ║
║                                                              ║
║   Log: "LSH index built: 50 items in 13 buckets (avg: 3.8)"  ║
╚══════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════╗
║  STEP 5 — Seed Demo User Profiles                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   user_sci_fi    → [item_2, item_3, item_4]                  ║
║                    Inception, Interstellar, The Matrix       ║
║                                                              ║
║   user_action    → [item_1, item_5, item_9]                  ║
║                    The Dark Knight, Endgame, Mad Max         ║
║                                                              ║
║   user_horror    → [item_11, item_12, item_29]               ║
║                    Get Out, Hereditary, Us                   ║
║                                                              ║
║   user_animation → [item_40, item_41, item_42]               ║
║                    Coco, Soul, Ratatouille                   ║
║                                                              ║
║   user_drama     → [item_38, item_39, item_6]                ║
║                    Portrait of a Lady on Fire,               ║
║                    Manchester by the Sea, Parasite           ║
║                                                              ║
║   Log: "Startup complete. 50 items, 5 demo profiles seeded." ║
╚══════════════════════════════════════════════════════════════╝
         │
         ▼
  Application ready — accepting requests on port 8080
```

---

## 5. Request Pipeline

```
GET /recommend/user_sci_fi
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 1 — Cache Lookup                                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   cacheService.get("user_sci_fi")                                    ║
║   → redis.opsForValue().get("rec:user_sci_fi")                       ║
║                                                                      ║
║   HIT? ────────────────────────────────────────▶  HTTP 200 (< 1 ms) ║
║   MISS ────────────────────────────────────────▶  continue           ║
╚══════════════════════════════════════════════════════════════════════╝
        │  (MISS — first request or TTL expired)
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 2 — Profile Lookup                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   userProfileRepository.findById("user_sci_fi")                      ║
║   → UserProfile("user_sci_fi", ["item_2","item_3","item_4"])          ║
║                                                                      ║
║   not found? ──────────────────────────────────▶  HTTP 404           ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 3 — Collect Viewed Vectors                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ingestionService.findById("item_2")  →  Inception    double[456]   ║
║   ingestionService.findById("item_3")  →  Interstellar double[456]   ║
║   ingestionService.findById("item_4")  →  The Matrix   double[456]   ║
║                                                                      ║
║   item not found or not vectorized?  →  skip with debug log          ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 4 — Build Preference Vector                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   preferenceVector[i] = (inception[i] + interstellar[i] + matrix[i])║
║                          ────────────────────────────────────────    ║
║                                          3                           ║
║                                                                      ║
║   → a single double[456] representing the user's "average taste"     ║
║     High values in dimensions for: dream, space, reality, mind, …    ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 5 — LSH Candidate Retrieval                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   hash = randomProjection.hash(preferenceVector)  →  "0110"          ║
║                                                                      ║
║   buckets["0110"] not empty?                                         ║
║     YES ──▶  candidates = [Arrival, Ex Machina, Blade Runner 2049]   ║
║      NO ──▶  expand to Hamming-distance-1 neighbors                  ║
║              "1110", "0010", "0100", "0111"  (flip each bit once)    ║
║              still empty? ──▶  full scan: candidates = all 50 items  ║
║                                                                      ║
║   all candidates already viewed?                                     ║
║     YES ──▶  engine falls back to full scan (small dataset edge case)║
║      NO ──▶  proceed                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 6 — Score and Filter                                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   for each candidate not in viewedSet:                               ║
║                                                                      ║
║     score = (preferenceVec · candidate.vec)                          ║
║             ─────────────────────────────────                        ║
║             ‖preferenceVec‖ × ‖candidate.vec‖                        ║
║                                                                      ║
║     RecommendationResult(id, name, description, score)               ║
║                                                                      ║
║   sort by score descending                                           ║
║   return results[0 .. min(topN, results.size)]                       ╙
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STEP 7 — Populate Cache                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   redis.opsForValue().set("rec:user_sci_fi", results, 10 minutes)    ║
║   (failure is non-fatal — next request recomputes)                   ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
  HTTP 200
  [
    { "id": "item_7",  "name": "Arrival",         "score": 0.847 },
    { "id": "item_18", "name": "Ex Machina",       "score": 0.792 },
    { "id": "item_22", "name": "Blade Runner 2049","score": 0.761 },
    ...
  ]
```

---

## 6. Core Algorithms — Deep Dive

### 6.1 TF-IDF Vectorization

**The problem:** Machine learning algorithms work on numbers, not text. We need to turn an item description like *"A mind-bending sci-fi thriller about dreams within dreams"* into a numeric vector that captures what the item is *about* — in a way that items about similar topics end up numerically close.

**Term Frequency (TF)** — how often a word appears in one document, normalized by document length:

```
  TF(term, doc) = count of term in doc
                  ───────────────────────
                  total tokens in doc

  "Inception" description has 25 tokens.
  "dream" appears 3 times.
  TF("dream", Inception) = 3/25 = 0.12
```

**Inverse Document Frequency (IDF)** — how rare a word is across all documents. Rare words carry more meaning:

```
  IDF(term) = log( (1 + N) / (1 + df(term)) )

  N       = total item count (50)
  df(term) = number of items containing that term

  "dream"   appears in 2 of 50 items  →  IDF = log(51/3) = 2.83  (rare, informative)
  "film"    appears in 45 of 50 items  →  IDF = log(51/46) = 0.10  (common, noise)
  "story"   appears in 30 of 50 items  →  IDF = log(51/31) = 0.50  (moderate)
```

**TF-IDF score** — the final weight for each term in each document:

```
  TFIDF(term, doc) = TF(term, doc) × IDF(term)

  "dream" in Inception  =  0.12 × 2.83  =  0.34   (strong signal)
  "film"  in Inception  =  0.08 × 0.10  =  0.008  (near noise)
```

**Vector construction:**

```
  Vocabulary (456 terms, ranked by IDF, highest first):
  index 0   "dream"       IDF=2.83
  index 1   "heist"       IDF=2.71
  index 2   "space"       IDF=2.60
  index 3   "parasite"    IDF=2.55
  …
  index 455  "character"   IDF=0.05

  Inception  → double[456]:
    [0.34, 0.00, 0.00, 0.00, …, 0.02, …]
      ▲                             ▲
   "dream"                     common term

  Interstellar → double[456]:
    [0.05, 0.00, 0.29, 0.00, …, 0.01, …]
      ▲           ▲
   weak          "space" strong

  Get Out → double[456]:
    [0.00, 0.00, 0.00, 0.00, …, 0.03, …]
    (no "dream", no "space")
```

Items about similar topics end up with non-zero values in the same dimensions — cosine similarity exploits this. Items about completely different topics have non-overlapping non-zero positions — their cosine similarity approaches 0.

---

### 6.2 Cosine Similarity

**The idea:** Two items are similar if their TF-IDF vectors point in the same *direction*, regardless of their magnitudes (description lengths).

```
  cosine_similarity(A, B) = ──────────────────
                              ‖A‖ × ‖B‖

  where:
    A · B  =  dot product  =  Σ(Aᵢ × Bᵢ)       (how much they overlap)
    ‖A‖    =  magnitude    =  √(Σ Aᵢ²)          (length of vector A)
    ‖B‖    =  magnitude    =  √(Σ Bᵢ²)          (length of vector B)

  ┌──────────────────────────────────────────────────────────────────────┐
  │   score = 1.0   ──  identical direction (same topic)                 │
  │   score = 0.5   ──  partially overlapping topics                     │
  │   score = 0.0   ──  orthogonal (zero shared vocabulary)              │
  │   Edge case: if ‖A‖ == 0 or ‖B‖ == 0  →  return 0.0  (no NaN)       │
  └──────────────────────────────────────────────────────────────────────┘
```

**Why not Euclidean distance?** A short Inception description and a long Inception novelization have vectors of very different magnitudes. Euclidean distance would call them dissimilar. Cosine similarity looks only at the *angle* — it normalizes magnitude away — so both correctly score ~1.0 against each other.

```
  EXAMPLE (5-dimension simplified)
  vocabulary: ["dream", "space", "horror", "comedy", "robot"]

  Inception:     [0.42, 0.00, 0.00, 0.00, 0.00]
  Interstellar:  [0.10, 0.55, 0.00, 0.00, 0.00]
  Get Out:       [0.00, 0.00, 0.61, 0.00, 0.00]

  cosine(Inception, Interstellar):
    dot = 0.42×0.10 + 0×0.55 = 0.042
    ‖Inception‖ = √(0.42²) = 0.42
    ‖Interstellar‖ = √(0.10² + 0.55²) = 0.559
    score = 0.042 / (0.42 × 0.559) = 0.179   ← some similarity (both have "dream"/"space")

  cosine(Inception, Get Out):
    dot = 0 (no shared non-zero dimensions)
    score = 0.0   ← no similarity
```

---

### 6.3 Locality-Sensitive Hashing (LSH)

**The scalability problem:** Scoring every item against every user's preference vector is O(N × D) per request (N = items, D = dimensions). With millions of items this is too slow.

**The insight:** We do not need the *exact* nearest neighbors. We need items that are *approximately* similar. LSH lets us skip most items while missing very few truly similar ones.

#### Random Projection — how items are hashed

Generate `k` random hyperplanes in the vector space (unit vectors with Gaussian random components). Each hyperplane divides the full vector space into two half-spaces.

```
  For a vector v and hyperplane h:

  bit = (v · h ≥ 0) ? '1' : '0'

  Do this for k=4 hyperplanes → 4-character binary string

  preferenceVector →  "0110"
  Inception vector →  "0110"   ← same bucket (similar direction)
  Get Out vector   →  "1001"   ← different bucket (different direction)
```

**Why this works:** If two vectors have high cosine similarity (small angle θ between them), they are likely to fall on the same side of each random hyperplane:

```
  P(same bit for one hyperplane) = 1 − θ/π

  θ = 10°  (very similar) → P = 1 − 10/180 = 0.944   → likely same bucket
  θ = 90°  (unrelated)    → P = 1 − 90/180 = 0.500   → coin-flip
  θ = 170° (opposite)     → P = 1 − 170/180 = 0.056  → almost surely different bucket
```

#### Bucket structure

```
  AFTER LSHIndex.build(items) with k=4, 50 items:

  hash "0000" → []
  hash "0001" → [The Dark Knight, Mad Max: Fury Road]
  hash "0010" → []
  hash "0011" → [Parasite, Portrait of a Lady on Fire]
  hash "0100" → [Coco, Soul, Ratatouille]
  hash "0101" → []
  hash "0110" → [Inception, Interstellar, Arrival]         ← 3 items
  hash "0111" → [Ex Machina, Blade Runner 2049]
  …
  hash "1111" → [Get Out, Hereditary, Us, The Babadook]

  At query time:
    preferenceVector (user_sci_fi) → "0110"
    candidates = [Inception, Interstellar, Arrival]
    Score only 3 items, not all 50.
```

#### Fallback strategy — what happens if the bucket is empty

```
  1. Exact bucket ("0110") not empty? ──────────────────▶ use it
  2. Exact bucket empty?
       expand to Hamming-distance-1 neighbors:
         "1110" (flip bit 0)
         "0010" (flip bit 1)
         "0100" (flip bit 2)
         "0111" (flip bit 3)
       collect items from all matching neighbor buckets
       not empty? ──────────────────────────────────────▶ use expanded set
  3. Still empty? ─────────────────────────────────────▶ full scan (all items)
```

#### Choosing k (num-hash-bits)

```
  Rule of thumb: 2^k ≈ N / 5   (aim for ~5 items per bucket)

  N=50   →  k=4  →  16 possible buckets  →  ~3.1 items/bucket  ✓
  N=50   →  k=12 →  4096 possible buckets →  every item in own bucket ✗

  k=12 with 50 items: the user's preference vector hashes to a bucket
  that almost certainly contains 0 items (4096 buckets for 50 items).
  The Hamming-1 expansion finds 1–2 items, all likely already viewed.
  Engine returns empty recommendations. This was a real bug — fixed by
  lsh.num-hash-bits=4 and the all-viewed fallback in RecommendationEngine.
```

#### Reproducibility — fixed seed

```
  RandomProjection uses seed=42L:

  rng = new Random(42L)
  plane[i] = Gaussian(rng)  // same sequence every JVM start

  Without a fixed seed: every restart generates different hyperplanes →
  different bucket assignments → Redis cache built before restart becomes
  invalid (cached results from wrong buckets). Fixed seed ensures bucket
  assignments are deterministic across restarts.
```

---

### 6.4 Cache-Aside Pattern

Every recommendation request runs TF-IDF lookups, LSH queries, and cosine scoring. The same user requesting recommendations twice inside the TTL window should not pay that cost twice.

```
  get(userId):
  ┌─────────────────────────────────────────────────────────────┐
  │   1. value = redis.get("rec:" + userId)                     │
  │   2. if value != null ──▶  return value           // HIT    │
  │   3. results = engine.recommend(userId)           // MISS   │
  │   4. redis.set("rec:" + userId, results, TTL)               │
  │   5. return results                                         │
  └─────────────────────────────────────────────────────────────┘

  put(userId, results):
  ┌─────────────────────────────────────────────────────────────┐
  │   redis.opsForValue().set("rec:" + userId, results, TTL)    │
  │   (TTL default: 10 minutes)                                 │
  └─────────────────────────────────────────────────────────────┘

  evict(userId):
  ┌─────────────────────────────────────────────────────────────┐
  │   redis.delete("rec:" + userId)                             │
  │   called by POST /users/{id}/items/{id} on every new view   │
  └─────────────────────────────────────────────────────────────┘
```

**When the cache is invalidated:** When `POST /users/{userId}/items/{itemId}` is called, the user's history changes. Their cached recommendations are stale. The cache entry is evicted immediately so the next GET recomputes with the updated history.

**Redis failure handling:** All three operations are wrapped in `try/catch`. A Redis outage degrades to no-cache mode (every request recomputes at full engine cost) but the API continues serving. No exception propagates to the client.

**Serialization:** `GenericJackson2JsonRedisSerializer` — results are stored as JSON so they survive application restarts.

---

## 7. Component Deep-Dive

### 7.1 DataIngestionService

**Package:** `com.recommendationengine.ingestion`

Reads `items.csv` from the classpath and maintains a thread-safe in-memory store.

```
  CSV row → Item(id, name, description)
                │
         ConcurrentHashMap<String, Item>
                │
    findById(id)     → Item (or null)
    getAllItems()     → unmodifiable Collection<Item>
    size()           → int

  ingest() clears the store before reloading.
  Malformed rows are logged as warnings and skipped — no crash on bad data.
```

---

### 7.2 TextPreprocessor

**Package:** `com.recommendationengine.nlp`

Converts raw text into clean tokens. Applied to every item description before TF-IDF scoring.

```
  Input:  "A mind-bending sci-fi thriller about dreams within dreams!"
          │
          ▼
  lowercase + strip non-alpha  →  "a mind bending sci fi thriller about dreams within dreams"
          │
          ▼
  split on whitespace  →  ["a","mind","bending","sci","fi","thriller","about","dreams","within","dreams"]
          │
          ▼
  filter: length > 1    →  remove "a"
  filter: not stop word →  remove "about", "within"
          │
          ▼
  Output: ["mind","bending","sci","fi","thriller","dreams","dreams"]
```

Stop words removed include: `a, an, the, is, are, was, were, be, have, has, do, does, will, would, could, should, may, in, on, at, to, for, of, with, by, from, and, but, or, so, that, this, which, who, it, its, they, their, i, you, we, …` (80+ terms)

---

### 7.3 TfIdfVectorizer

**Package:** `com.recommendationengine.nlp`

Stateful vectorizer — must be fitted before it can transform.

```
  STATE
  ─────
  vocabulary:  Map<String, Integer>  (term → index in vector)
  idfScores:   Map<String, Double>   (term → precomputed IDF)
  fitted:      boolean

  fit(List<Item>):
  ┌──────────────────────────────────────────────────────────────────┐
  │   for each item → tokenize description → count unique terms      │
  │   docFrequency[term] = # items containing term                   │
  │                                                                  │
  │   for each term:                                                 │
  │     IDF = log( (N+1) / (df+1) )                                  │
  │                                                                  │
  │   sort by IDF descending → keep top maxVocabSize                 │
  │   → vocabulary + idfScores set  →  fitted = true                 │
  └──────────────────────────────────────────────────────────────────┘

  transform(Item) → double[vocabSize]:
  ┌──────────────────────────────────────────────────────────────────┐
  │   if !fitted → throw IllegalStateException                       │
  │   tokens = preprocessor.tokenize(item.description)              │
  │   count TF for each token                                        │
  │   for each term in vocabulary at index i:                        │
  │     vector[i] = TF(term, item) × IDF(term)                       │
  │   return vector  (length = vocabulary.size())                    │
  └──────────────────────────────────────────────────────────────────┘

  fitAndTransform(List<Item>):
    fit(items)
    for each item: item.setVector(transform(item))
```

**Key implementation note:** `maxVocabSize` has a field-level default (`= 500`) in addition to the `@Value` annotation. Without this, unit tests that construct the class with `new TfIdfVectorizer(preprocessor)` — where Spring's `@Value` injection does not run — would get `maxVocabSize = 0` and produce empty vocabularies.

---

### 7.4 RandomProjection

**Package:** `com.recommendationengine.index`

Generates the random hyperplanes used for LSH hashing. Must be called once before hashing any vectors.

```
  initialize(int k, int dim):
  ┌──────────────────────────────────────────────────────────────────┐
  │   rng = new Random(42L)          // fixed seed → reproducible    │
  │   for i in 0..k-1:                                               │
  │     plane[i] = new double[dim]                                   │
  │     for j in 0..dim-1:                                           │
  │       plane[i][j] = rng.nextGaussian()                           │
  │     normalize plane[i] to unit length                            │
  └──────────────────────────────────────────────────────────────────┘

  hash(double[] vector) → String:
  ┌──────────────────────────────────────────────────────────────────┐
  │   char[] bits = new char[k]                                      │
  │   for i in 0..k-1:                                               │
  │     dot = Σ( vector[j] × plane[i][j] )                           │
  │     bits[i] = (dot ≥ 0) ? '1' : '0'                              │
  │   return new String(bits)    // e.g. "0110" for k=4              │
  └──────────────────────────────────────────────────────────────────┘
```

Why normalize each hyperplane to unit length? The sign of the dot product is what matters for hashing, not the magnitude. Normalization ensures each hyperplane has equal "voting power" and the distribution of buckets is balanced.

---

### 7.5 LSHIndex

**Package:** `com.recommendationengine.index`

Builds the bucket map and answers candidate queries.

```
  STATE
  ─────
  buckets:  Map<String, List<Item>>  (hash string → items)
  allItems: List<Item>               (full fallback set)

  build(List<Item>):
  ┌──────────────────────────────────────────────────────────────────┐
  │   projection.initialize(numHashBits, vectorDimensions)           │
  │   for each item:                                                 │
  │     hash = projection.hash(item.vector)                          │
  │     buckets.computeIfAbsent(hash, k -> new ArrayList<>()).add()  │
  └──────────────────────────────────────────────────────────────────┘

  getCandidates(double[] queryVector) → List<Item>:
  ┌──────────────────────────────────────────────────────────────────┐
  │   hash = projection.hash(queryVector)                            │
  │                                                                  │
  │   bucket = buckets.get(hash)                                     │
  │   if bucket not empty  ──────────────────────▶  return bucket    │
  │                                                                  │
  │   expanded = expandedSearch(hash)                                │
  │   if expanded not empty  ────────────────────▶  return expanded  │
  │                                                                  │
  │   ──────────────────────────────────────────▶  return allItems   │
  └──────────────────────────────────────────────────────────────────┘

  expandedSearch(String hash):
  ┌──────────────────────────────────────────────────────────────────┐
  │   for i in 0..hash.length-1:                                     │
  │     flip bit i  →  neighbor hash                                 │
  │     add buckets.get(neighborHash) to results                     │
  └──────────────────────────────────────────────────────────────────┘
```

---

### 7.6 RecommendationEngine

**Package:** `com.recommendationengine.engine`

Orchestrates the full recommendation pipeline.

```
  recommend(UserProfile profile, int topN):

  ┌──────────────────────────────────────────────────────────────────┐
  │  viewedIds = profile.viewedItemIds()                             │
  │  if empty → return [] (warn: no history)                         │
  └──────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  for each id in viewedIds:                                       │
  │    item = ingestionService.findById(id)                          │
  │    if item != null && item.getVector() != null:                  │
  │      viewedVectors.add(item.getVector())                         │
  │  if viewedVectors.isEmpty() → return [] (warn: no vectors)       │
  └──────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  preferenceVector = average(viewedVectors)                       │
  │    element-wise mean across all viewed item vectors              │
  └──────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  candidates = lshIndex.getCandidates(preferenceVector)           │
  │                                                                  │
  │  allViewed = candidates.stream().allMatch(viewedSet::contains)   │
  │  if allViewed:                                                   │
  │    candidates = ingestionService.getAllItems()  // fallback       │
  └──────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  for each candidate not in viewedSet:                            │
  │    if candidate.getVector() == null: skip                        │
  │    score = cosineSimilarity.compute(preferenceVector,            │
  │                                     candidate.getVector())       │
  │    results.add(new RecommendationResult(id, name, desc, score))  │
  │                                                                  │
  │  results.sort(by score descending)                               │
  │  return results[0 .. min(topN, results.size)]                    │
  └──────────────────────────────────────────────────────────────────┘
```

---

### 7.7 RecommendationCacheService

**Package:** `com.recommendationengine.cache`

Wraps Redis with the cache-aside pattern. All operations are non-fatal.

```
  get(String userId) → List<RecommendationResult> or null:
  ┌──────────────────────────────────────────────────────────────────┐
  │   try:                                                           │
  │     return (List) redis.opsForValue().get("rec:" + userId)       │
  │   catch Exception:                                               │
  │     log warning → return null  (treat as MISS)                   │
  └──────────────────────────────────────────────────────────────────┘

  put(String userId, List<RecommendationResult> results):
  ┌──────────────────────────────────────────────────────────────────┐
  │   try:                                                           │
  │     redis.opsForValue().set("rec:" + userId, results, TTL)       │
  │   catch Exception:                                               │
  │     log warning  (non-fatal: next request recomputes)            │
  └──────────────────────────────────────────────────────────────────┘

  evict(String userId):
  ┌──────────────────────────────────────────────────────────────────┐
  │   try:                                                           │
  │     redis.delete("rec:" + userId)                                │
  │   catch Exception:                                               │
  │     log warning                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

**Implementation note:** The constructor takes `RedisOperations<String, Object>` (interface) rather than the concrete `RedisTemplate` class. On Java 25, Mockito with byte-buddy cannot subclass `RedisTemplate` without experimental mode enabled. Using the interface avoids a dynamic subclassing requirement entirely and makes tests simpler.

---

### 7.8 UserProfileRepository

**Package:** `com.recommendationengine.user`

Thread-safe in-memory store for user profiles.

```
  STATE
  ─────
  store: ConcurrentHashMap<String, UserProfile>

  addViewedItem(String userId, String itemId):
  ┌──────────────────────────────────────────────────────────────────┐
  │   store.compute(userId, (id, existing) -> {                      │
  │     if existing == null:                                         │
  │       return new UserProfile(userId, List.of(itemId))            │
  │     list = new ArrayList(existing.viewedItemIds())               │
  │     if !list.contains(itemId): list.add(itemId)  // no dups      │
  │     return new UserProfile(userId, list)                         │
  │   })                                                             │
  └──────────────────────────────────────────────────────────────────┘
```

`ConcurrentHashMap.compute()` is atomic — no lost updates under concurrent requests. `UserProfile` is a Java record (immutable), so each update replaces the old record entirely.

---

### 7.9 RecommendationController

**Package:** `com.recommendationengine.controller`

```
  GET /recommend/{userId}:
  ┌──────────────────────────────────────────────────────────────────┐
  │   cached = cacheService.get(userId)                              │
  │   if cached != null → 200 OK (HIT)                               │
  │                                                                  │
  │   profile = userProfileRepository.findById(userId)               │
  │   if profile.isEmpty() → 404 Not Found                           │
  │                                                                  │
  │   results = engine.recommend(profile.get())                      │
  │   cacheService.put(userId, results)                              │
  │   → 200 OK                                                       │
  └──────────────────────────────────────────────────────────────────┘

  POST /users/{userId}/items/{itemId}:
  ┌──────────────────────────────────────────────────────────────────┐
  │   userProfileRepository.addViewedItem(userId, itemId)            │
  │   cacheService.evict(userId)                                     │
  │   → 200 OK                                                       │
  └──────────────────────────────────────────────────────────────────┘

  DELETE /cache/{userId}:
  ┌──────────────────────────────────────────────────────────────────┐
  │   cacheService.evict(userId)                                     │
  │   → 204 No Content                                               │
  └──────────────────────────────────────────────────────────────────┘
```

---

### 7.10 ApplicationStartupRunner

**Package:** `com.recommendationengine.startup`

Implements `CommandLineRunner` — executes once after the Spring context is fully initialized. Orchestrates the full startup pipeline and seeds demo data.

```
  run():
    1. DataIngestionService.ingest("/data/items.csv")
    2. TfIdfVectorizer.fitAndTransform(items)
    3. LSHIndex.build(items)
    4. seed 5 UserProfiles  (see startup pipeline above)
```

---

## 8. API Reference

### GET `/recommend/{userId}`

Returns top-N ranked recommendations for the user. Uses cache-aside — first call computes, subsequent calls within TTL return instantly.

**Path parameter:** `userId` — must match a known user profile

**Response 200:**
```json
[
  {
    "id": "item_7",
    "name": "Arrival",
    "description": "A linguist works with the military to communicate with alien lifeforms...",
    "score": 0.847
  },
  {
    "id": "item_18",
    "name": "Ex Machina",
    "description": "A young programmer is selected to participate in a ground-breaking experiment...",
    "score": 0.792
  }
]
```

**Response 404:** User not found

---

### POST `/users/{userId}/items/{itemId}`

Records that a user viewed an item. Creates the user profile on first call. Evicts cached recommendations so the next GET recomputes with the updated history.

**Response 200:** (empty body)

---

### DELETE `/cache/{userId}`

Manually evicts cached recommendations. Useful for forcing a recompute without changing the user's history.

**Response 204:** (empty body)

---

## 9. Configuration Reference

All configuration lives in `src/main/resources/application.properties`:

| Property | Default | Description |
|---|---|---|
| `server.port` | `8080` | HTTP port |
| `spring.data.redis.host` | `localhost` | Redis hostname |
| `spring.data.redis.port` | `6379` | Redis port |
| `recommendation.top-n` | `10` | Maximum recommendations returned per request |
| `recommendation.cache.ttl-minutes` | `10` | Redis TTL for cached results |
| `lsh.num-hash-bits` | `4` | Number of LSH hash bits (k). Rule: `2^k ≈ N/5` |
| `lsh.vector-dimensions` | `500` | Maximum TF-IDF vocabulary size |

**Tuning `lsh.num-hash-bits` for your dataset:**

| Item count | Recommended k | Buckets | Avg bucket size |
|---|---|---|---|
| 50 | 4 | 16 | ~3 |
| 500 | 6 | 64 | ~8 |
| 5 000 | 9 | 512 | ~10 |
| 50 000 | 13 | 8 192 | ~6 |

Too few bits → large buckets → more cosine scoring per request, but better recall.
Too many bits → tiny buckets → faster but misses similar items, or falls back to full scan.

---

## 10. Data Format

Items are loaded from `src/main/resources/data/items.csv`:

```csv
id,name,description
item_1,The Dark Knight,"When the menace known as the Joker wreaks havoc on Gotham..."
item_2,Inception,"A thief who steals corporate secrets through dream-sharing technology..."
```

| Column | Type | Description |
|---|---|---|
| `id` | String | Unique identifier. Used in API path params and user profiles. |
| `name` | String | Display name returned in recommendation results. |
| `description` | String | Free text. This is what gets vectorized. Richer descriptions produce better results. |

To use your own data: replace `items.csv` following this schema and restart. The vectorizer, LSH index, and demo profiles all rebuild automatically.

---

## 11. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Java 17 (source/target), Java 25 JVM | LTS stability; records for immutable models |
| Framework | Spring Boot 3.2.5 | Dependency injection, REST, auto-configuration |
| Cache | Redis 7 via Spring Data Redis | Sub-millisecond key lookups, configurable TTL |
| CSV parsing | OpenCSV 5.9 | Robust handling of quoted fields and multi-line descriptions |
| Build | Gradle 8 | Dependency management, test JVM args for Java 25 compatibility |
| Testing | JUnit 5, Mockito 5, Testcontainers | Unit + integration tests without a live Redis instance |
| Logging | SLF4J + Logback (via Spring Boot) | Structured logging at every pipeline stage |

> **Note on Lombok:** Lombok is intentionally excluded. It is incompatible with Java 25 (`NoSuchFieldException: TypeTag :: UNKNOWN` during annotation processing). Java records cover immutable models (`UserProfile`, `RecommendationResult`); plain getters/setters cover the mutable `Item` class.

---

## 12. Project Structure

```
src/
├── main/
│   ├── java/com/recommendationengine/
│   │   ├── RecommendationEngineApplication.java   ← Spring Boot entry point
│   │   ├── model/
│   │   │   ├── Item.java                          ← mutable: id, name, description, vector
│   │   │   ├── UserProfile.java                   ← record: userId, viewedItemIds
│   │   │   └── RecommendationResult.java          ← record: id, name, description, score
│   │   ├── nlp/
│   │   │   ├── TextPreprocessor.java              ← tokenize, stop-word removal
│   │   │   └── TfIdfVectorizer.java               ← fit() + transform() → double[]
│   │   ├── similarity/
│   │   │   └── CosineSimilarity.java              ← (A·B) / (‖A‖ × ‖B‖)
│   │   ├── ingestion/
│   │   │   └── DataIngestionService.java          ← CSV → ConcurrentHashMap
│   │   ├── index/
│   │   │   ├── RandomProjection.java              ← k random hyperplanes, seed=42
│   │   │   └── LSHIndex.java                      ← build + query hash buckets
│   │   ├── engine/
│   │   │   └── RecommendationEngine.java          ← orchestrate: vectors→LSH→cosine→sort
│   │   ├── cache/
│   │   │   └── RecommendationCacheService.java    ← Redis cache-aside wrapper
│   │   ├── user/
│   │   │   └── UserProfileRepository.java         ← in-memory, thread-safe
│   │   ├── controller/
│   │   │   └── RecommendationController.java      ← REST endpoints
│   │   ├── config/
│   │   │   └── RedisConfig.java                   ← RedisTemplate bean
│   │   └── startup/
│   │       └── ApplicationStartupRunner.java      ← ingest → vectorize → index → seed
│   └── resources/
│       ├── application.properties
│       └── data/items.csv                         ← 50 movies
└── test/
    ├── java/com/recommendationengine/
    │   ├── model/           ItemTest, UserProfileTest, RecommendationResultTest
    │   ├── nlp/             TextPreprocessorTest, TfIdfVectorizerTest
    │   ├── similarity/      CosineSimilarityTest
    │   ├── ingestion/       DataIngestionServiceTest
    │   ├── index/           RandomProjectionTest, LSHIndexTest
    │   ├── engine/          RecommendationEngineTest
    │   ├── cache/           RecommendationCacheServiceTest
    │   ├── controller/      RecommendationControllerTest
    │   └── startup/         ApplicationStartupRunnerTest
    └── resources/
        └── application-test.properties
```

---

## 13. Building and Running

**Prerequisites:** Java 17+, Docker Desktop (for Redis), Gradle wrapper included.

```bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis:7

# Build
./gradlew build

# Run
./gradlew bootRun
```

Expected startup log:
```
INFO  DataIngestionService       : Ingested 50 items from /data/items.csv
INFO  TfIdfVectorizer            : TF-IDF fitted on 50 documents — vocabulary size: 456
INFO  TfIdfVectorizer            : TF-IDF vectors computed for 50 items (dimensions: 456)
INFO  LSHIndex                   : LSH index built: 50 items in 13 buckets (avg size: 3.8)
INFO  ApplicationStartupRunner   : Startup complete. 50 items, 5 demo profiles seeded.
```

```bash
# Get recommendations for the sci-fi demo user
curl http://localhost:8080/recommend/user_sci_fi

# Record a view (creates user_new on first call)
curl -X POST http://localhost:8080/users/user_new/items/item_1

# Get recommendations based on that view
curl http://localhost:8080/recommend/user_new

# Manually evict cache
curl -X DELETE http://localhost:8080/cache/user_sci_fi
```

**Demo user profiles:**

| User ID | Viewed Items | Theme |
|---|---|---|
| `user_sci_fi` | Inception, Interstellar, The Matrix | Sci-fi, mind-bending |
| `user_action` | The Dark Knight, Avengers: Endgame, Mad Max | Action, superhero |
| `user_horror` | Get Out, Hereditary, Us | Horror, psychological |
| `user_animation` | Coco, Soul, Ratatouille | Animation, Pixar |
| `user_drama` | Portrait of a Lady on Fire, Manchester by the Sea, Parasite | Drama |

```bash
# Run all tests (no live Redis required)
./gradlew test

# Run a specific test class
./gradlew test --tests "com.recommendationengine.engine.RecommendationEngineTest"
```

---

## 14. Test Coverage

**97 tests — all passing.**

| Test Class | Type | What it covers |
|---|---|---|
| `ItemTest` | Unit | Getters, setters, vector state transitions |
| `UserProfileTest` | Unit | Record equality, immutability |
| `RecommendationResultTest` | Unit | Record construction, score field |
| `TextPreprocessorTest` | Unit | Tokenization, stop-word removal, null/blank edge cases |
| `TfIdfVectorizerTest` | Unit | fit/transform, IDF math, vocabulary capping, state guards |
| `CosineSimilarityTest` | Unit | Known vectors, orthogonal vectors, identical, zero vector |
| `DataIngestionServiceTest` | Unit | CSV parsing, findById, size, re-ingest |
| `RandomProjectionTest` | Unit | Hash determinism (fixed seed), bit count, initialization guard |
| `LSHIndexTest` | Unit | Bucket construction, exact/Hamming/fallback lookup |
| `RecommendationEngineTest` | Unit | Full pipeline on real CSV, viewed filtering, top-N, empty history |
| `RecommendationCacheServiceTest` | Unit | HIT/MISS/evict flows, Redis failure non-fatal |
| `RecommendationControllerTest` | Web (MockMvc) | HTTP status codes, 404 on unknown user, endpoint wiring |
| `ApplicationStartupRunnerTest` | Integration | Full Spring context, LSH built, all items vectorized, demo profiles seeded |

**Java 25 test compatibility** requires these JVM args in `build.gradle`:

```gradle
test {
    jvmArgs '-Dnet.bytebuddy.experimental=true',
            '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
            '--add-opens', 'java.base/java.lang.reflect=ALL-UNNAMED',
            '--add-opens', 'java.base/java.util=ALL-UNNAMED'
}
```

The integration test (`ApplicationStartupRunnerTest`) uses `@MockBean(RedisConnectionFactory)` so no live Redis instance is needed in CI. All other tests use Mockito mocks or operate entirely in memory.

---

## 15. Design Decisions & Trade-offs

**Why content-based filtering instead of collaborative filtering?**
Collaborative filtering requires a meaningful user-item interaction matrix — thousands of users with substantial overlap in viewed items. With a fresh dataset and few users, the matrix is too sparse to find reliable user-to-user similarity. Content-based filtering works from day one with a single user.

**Why `byte[]`-agnostic? — Actually, why not.** Item descriptions are UTF-8 strings and so are keys. String types are used throughout. A general-purpose storage engine (like the LSM-Engine) needs byte-agnostic interfaces; a domain-specific recommendation engine does not.

**Why `ConcurrentHashMap` for user profiles?**
`addViewedItem` uses `compute()`, which is atomic at the key level — no external lock needed for concurrent view events from the same user. Profile immutability (`UserProfile` is a record) means reads never see partial state.

**Why `ConcurrentSkipListMap` is NOT used for the MemTable equivalent here (DataIngestionService)?**
Items are write-once (loaded at startup), never updated, and always read. A `ConcurrentHashMap` gives O(1) reads by ID, which is all the engine needs. A skip list adds sorted iteration at the cost of O(log n) inserts for no benefit here.

**Why fixed seed=42 for RandomProjection?**
Without a fixed seed, every application restart generates different hyperplanes and different bucket assignments. A cache entry written in one session would be replayed against incorrect buckets after restart. Fixed seed makes bucket assignments deterministic and cache entries valid across restarts.

**Why sparse averaging for the preference vector instead of something more sophisticated?**
Simple averaging is interpretable, fast, and sufficient for content-based filtering over a small-to-medium catalogue. It does give equal weight to all viewed items regardless of recency or frequency. A more sophisticated approach (recency-weighted average, attention mechanism, or a learned embedding) would improve recommendation quality but at far greater complexity. Averaging is the right baseline to build on.

**Why `lsh.num-hash-bits=4` and not the default 12?**
With 50 items and 12 bits, the index creates up to 4096 possible buckets. Each item hashes to its own unique bucket. The user's preference vector hashes to a bucket with 0 items. The Hamming-1 expansion finds at most a few items. The engine falls back to full scan — which works but defeats the purpose of LSH. The rule of thumb `2^k ≈ N/5` gives `k=4` for 50 items: 16 buckets, ~3 items each. The engine finds real candidates immediately.

**Why Redis failures are non-fatal?**
The cache is an optimization, not a correctness requirement. A Redis outage should degrade performance (every request recomputes), not degrade availability (500 errors). The try/catch wrappers ensure the recommendation API keeps serving even when the cache layer is completely down.

**Why `@MockBean(RedisConnectionFactory)` in integration tests instead of `@MockBean(RedisTemplate)`?**
Spring's `redisKeyValueAdapter` requires a concrete `RedisTemplate` bean internally. Mocking `RedisTemplate` with `@MockBean` replaces the bean with a Mockito proxy that does not satisfy this internal dependency, causing a `ClassCastException` at context startup. Mocking `RedisConnectionFactory` instead lets Spring construct a real `RedisTemplate` bean wired to a mock connection — no live Redis required, no internal Spring conflicts.
