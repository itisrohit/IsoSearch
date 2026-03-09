# Technical Analysis & Benchmarks

## Overview
This document analyzes the retrieval performance and computational efficiency of the **IsoSearch** vector search pipeline against a standard "General RAG" architecture that utilizes native flat exact k-Nearest Neighbors (KNN) search over dense Euclidean space. 

IsoSearch achieves its latency and computational cost reductions by transforming dense 384-dimensional semantic embeddings (e.g., HuggingFace BGE models) into compact 64-bit quantized binary fingerprints. These fingerprints are then indexed via Hash Table bucket intersections and traversed efficiently using a Hierarchical Navigable Small World (HNSW) graph in Hamming space.

## Hardware Configuration
All benchmarks were executed locally on consumer hardware to establish a baseline.

* **Device:** Apple MacBook Pro (13-inch, M1, 2020)
* **Chip:** Apple M1 Silicon (ARM64)
* **Cores:** 8 Cores (4 Performance & 4 Efficiency)
* **Memory:** 8 GB Unified Memory
* **Language Runtime:** Rust Edition 2024 (optimized with `opt-level = 3`, LTO enabled)
* **Benchmarking Tool:** `criterion` framework

---

## 1. Pipeline Breakdown (Simulated 10,000 Document Index)

The full retrieval pipeline scales across four primary phases. Measurements below represent the average latency required to process a *single* dense 384-dimensional query vector through the pipeline on a simulated index of 10,000 dummy documents. Note that these are preliminary micro-benchmarks and real-world performance with fully populated data may vary.

### Phase 1: Math Operations
* **Operations:** Whitening Normalization (Mean Centering & Covariance Scaling) + Poincaré Ball Projection (Hyperbolic Mapping) + Johnson-Lindenstrauss Gaussian Random Projection (384D -> 128D).
* **Time:** ~14.2 µs
* **Analysis:** Rust's optimized `ndarray` operations make projecting the query vector across multiple transformation spaces highly efficient.

### Phase 2: Hashing Pipeline
* **Operations:** Locality-Sensitive Hashing (LSH) via SimHash (Sign-Random Projection) + Binary Quantization (packing bits into `u64`).
* **Time:** ~0.87 µs
* **Analysis:** Condensing the reduced dense vector into an exact integer bitset executes extremely fast.

### Phase 3: Index Lookup
* **Operations:** Bucket Intersection over a simulated index.
* **Time:** ~279.3 µs
* **Analysis:** The majority of the pipeline's runtime is spent accessing memory during bucketing. Using Hash Tables helps to keep this lookup cost manageable.

### Phase 4: Hamming Search
* **Operations:** Target nearest-neighbor candidate traversal using a greedy algorithm over an HNSW graph in Hamming Space (calculated via bitwise XOR & popcount operations).
* **Time:** ~0.11 µs
* **Analysis:** Computing binary string overlaps against the candidate set is highly optimized by utilizing native CPU bitwise operations.

---

## 2. IsoSearch vs. Baseline (10k Simulated Documents)

General Retrieval-Augmented Generation (RAG) models often rely on exact "flat" vector databases without quantization, computing exact Euclidean distances linearly against every chunk.

On our 10,000-document simulated set, the baseline is calculated:

| System Config | Setup | Average Retrieval Time | Speed Multiplier |
| :--- | :--- | :--- | :--- |
| **General RAG Baseline** | Dense `384D f32` scanning via Exact Euclidean Search | ~946.0 µs | 1.0x |
| **IsoSearch** | Quantized `u64` pipeline via Bucket Intersection & HNSW | ~294.4 µs | ~3.2x Faster |

### Potential Scalability Benefits
While the benchmarks above only measure 10,000 simulated documents, the mathematical properties of IsoSearch's operations hint at stronger theoretical scalability compared to naive flat-search architectures:

1.  **Linear vs. Logarithmic:** General explicit search scales linearly $O(N)$. IsoSearch utilizes bucket hashing $O(1)$ and hierarchical graph traversals that generally approximate $O(\log N)$ or bounded linear-scaling factors.
2.  **Memory Footprint:** By relying on `u64` representations, the index storage size per vector scales radically downward compared to storing raw `f32` vectors, allowing significantly more vectors to be kept in physical RAM.

*Note: Future experiments are required to validate these multi-million document boundaries in production settings with actual embeddings.*
