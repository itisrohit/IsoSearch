# Technical Specification: IsoSearch Retrieval System

## Background & Motivation (The Market Gap)

In the current landscape of Retrieval-Augmented Generation (RAG) and semantic search, standard vector databases (such as Pinecone, Milvus, Qdrant, and FAISS) typically index raw high-dimensional `f32` (floating point) embeddings. While highly accurate, using exact Euclidean or Cosine similarity over such vectors introduces massive bottlenecks:
1. **Hardware Costs:** Storing hundreds of millions of dense 384-dimensional or 768-dimensional vectors requires exorbitant amounts of expensive high-speed RAM.
2. **Compute Limits:** Scanning these vectors—even algorithmically via standard HNSW structures—relies heavily on floating point (`f32`) dot-product math. This scales poorly as datasets grow, often necessitating expensive multi-GPU enterprise infrastructure to maintain responsive latencies.

**IsoSearch is an experimental project attempting to circumvent this scaling barrier.** 
Instead of tackling massive vector scale with brute-force hardware, IsoSearch proposes a highly mathematical transformation pipeline. By passing semantic vectors sequentially through Whitening, Hyperbolic Geometry limits (Poincaré Ball), Random Projections (Johnson-Lindenstrauss lemma), and Locality-Sensitive Hashing, IsoSearch aims to compress expansive dimensional vectors into a highly compact **64-bit integer (`u64`)**.

The theoretical necessity for this experiment lies in changing the database economics: if successful, memory footprint shrinks by ~95%, and compute operations transition from expensive multi-dimensional floating point math to natively hardware-accelerated, single-cycle XOR and popcount bitwise operations in strict Hamming Space. This experiment aims to determine if extreme latency reduction ($O(1)$ Hash lookups and $O(\log N)$ bitwise graph traversals) can be achieved on cheap, commodity CPUs without fatally destroying search recall.

## Objectives and Performance Targets

The primary objective is the development of a **highly optimized approximate nearest neighbor (ANN) retriever** capable of indexing and searching document collections with minimal computational overhead.

| Metric | Target Specification |
| :--- | :--- |
| **Search Latency** | Optimized per query (Target: Minimal latency vs exact flat search) |
| **Dataset Scale** | Designed to scale for large document datasets |
| **Index Memory Footprint** | Optimized for minimal size vs dense storage |
| **Retrieval Accuracy** | Goal: Retain high Recall relative to exact search |
| **Inference Hardware** | CPU-optimized execution (no GPU requirement) |

---

## Architectural Pipeline Overview

The retrieval process follows a multi-stage optimization pipeline designed to minimize computational complexity at each successive step:

1. **[X] Routing Network/Partitioning** (Search Space Reduction) - *Implemented KMeansRouter*
2. **[X] Feature Extraction** (Embedding Generation) - *Implemented HuggingFaceEmbedder with BAAI/bge-small-en-v1.5*
3. **[X] Geometric Normalization** (Mean Centering) - *Implemented WhiteningNormalizer*
4. **[X] Spectral Decorrelation** (Whitening) - *Implemented SVD-based Covariance Whitening*
5. **[X] Hyperbolic Mapping** (Poincaré Ball Projection) - *Implemented PoincareProjector*
6. **[X] Dimensionality Reduction** (Johnson–Lindenstrauss Random Projection) - *Implemented RandomProjector*
7. **[X] Locality-Sensitive Hashing (LSH)** - *Implemented SimHasher (SimHash)*
8. **[X] Binary Quantization** - *Implemented BinaryQuantizer (u64 packing)*
9. **[X] Bucket Intersection** - *Implemented BucketIndex for candidate retrieval*
10. **[X] HNSW Graph Traversal** (Hamming Space Search) - *Implemented HNSWGraph (greedy search)*
11. **[X] Precision Rescoring** (Full Vector Reranking) - *Implemented VectorStore rescoring*

---

## Detailed Pipeline Implementation

### 1. Partition Routing
To reduce the initial search space from $O(N)$ to $O(N/k)$, a lightweight routing model (e.g., MLP, Decision Tree, or K-Means classifier) predicts the most relevant index partition. This effectively narrows the search scope from millions of documents to a subset of approximately 200,000 candidates.
* **Latency Overhead:** ~0.1 ms.

### 2. Semantic Embedding Generation
High-dimensional semantic vectors (384D) are generated using the **HuggingFace Inference API** with the `BAAI/bge-small-en-v1.5` model. This provides free, serverless embedding generation without requiring local GPU compute.
* **Implementation:** `HuggingFaceEmbedder` struct with authentication via `HF_TOKEN`
* **Endpoint:** `https://router.huggingface.co/hf-inference/models/{model_id}`
* **Embedding Dimension:** 384D (optimized for efficiency)

### 3. Geometric Normalization (Mean Centering)
Elimination of global bias to ensure the embedding distribution is centered around the origin.
* **Operation:** $x' = x - \mu$ (where $\mu$ is the dataset mean).

### 4. Anisotropy Correction (Whitening)
Normalization of variance across dimensions to improve similarity discrimination and mitigate anisotropy commonly found in transformer-based embeddings.
* **Operation:** $x' = \Lambda^{-1/2} U^T (x - \mu)$ (via Eigen-decomposition).

### 5. Hyperbolic Space Projection
Mapping vectors into a non-Euclidean Poincaré ball to superiorly represent latent hierarchical semantic relationships.
* **Operation:** $x_h = \tanh(\|x\|) \cdot \frac{x}{\|x\|}$.

### 6. Dimensionality Reduction (Random Projection)
Compression of the vector space (e.g., 768D → 128D) while preserving pairwise distances according to the Johnson–Lindenstrauss lemma.
* **Operation:** $y = R x$ (where $R$ is a sparse random matrix).

### 7. Locality-Sensitive Hashing (LSH)
Transformation of continuous vectors into discrete hash codes for rapid approximate matching.
* **Operation:** $h = \text{sign}(w \cdot x)$. Each hyperplane intersection generates 1 bit of information.

### 8. Binary Quantization
Final compression of high-dimensional projections into efficient bitstreams (e.g., 128 bits resulting in 16 bytes per vector).

### 9. Bucket Prioritization
LSH-generated bitstrings are utilized to identify candidate buckets for inspection, further reducing the retrieval candidates to approximately $O(10^3)$.

### 10. HNSW Search in Hamming Space
Hierarchical Navigable Small World (HNSW) graph traversal within the selected buckets using Hamming distance for high-speed nearest neighbor identification.
* **Complexity:** $O(\log N)$.

### 11. Full-Precision Reranking
Final rescoring of the top-K candidates (e.g., top 100) using original high-precision embeddings and cosine similarity to ensure target recall.

---

## System Architecture

```text
       [Query]
          |
    [Routing Stage] -> Index Partition Selection
          |
  [Embedding Stage] -> Transformer Inference
          |
[Normalization Stage] -> Mean Centering & Whitening
          |
  [Projection Stage] -> Hyperbolic & Random Projection
          |
    [Quantization] -> LSH/Binary Encoding
          |
   [Indexing Stage] -> HNSW Entry Point
          |
  [Hamming Retrieval] -> Top Candidate Selection
          |
  [Precision Rerank] -> Full Vector Similarity
          |
       [Results]
```

---

## Technical Implementation Stack (Rust)

The system will be implemented natively in **Rust** to leverage its strict memory safety guarantees and efficient handling of high-concurrency workloads.

## Technical Implementation Stack (Rust)

| Component | Current Implementation | Status |
| :--- | :--- | :--- |
| **Linear Algebra** | `ndarray` with `ndarray-linalg` (OpenBLAS/LAPACK backend) | ✅ Implemented |
| **Inference Engine** | API-based client (`reqwest`, `tokio`) connecting to HuggingFace | ✅ Implemented |
| **Graph Indexing** | Native Rust HNSW (Greedy Base-layer Search) | ✅ Implemented |
| **Core Engine SIMD** | Explicit `NEON` (ARM) and `AVX2` (x86_64) Intrinsic Kernels + LLVM Auto-vectorization | ✅ Optimized |
| **Concurrency** | `Rayon` for multi-threaded projection with default work-stealing scheduler | ✅ Optimized |
| **Serialization** | `serde` and `bincode` with buffered I/O for high-speed persistence | ✅ Optimized |
| **Cache-Locality** | Memory-aligned allocation for HNSW graph nodes | ⏳ Pending |

---

## Current Status & Future Goals

### Milestones Reached
Every technical stage defined in the **Architectural Pipeline Overview** (Steps 1 through 11) has been successfully implemented natively in Rust. The end-to-end mathematical framework is fully operational—capable of routing, transforming, hashing, quantizing, and traversing candidates in bounded Hamming space before applying an exact precision rescore.

For the initial hardware micro-benchmarks regarding execution latency and $O(1)/O(\log N)$ scalability observations compared to traditional "General RAG" systems, please consult the [Technical Analysis & Benchmarks](./technical_analysis.md) report.

### Current Focus: Phase 5 — Recall Evaluation & Accuracy Tuning
We are currently shifting focus from pure systems optimization to **mathematical verification**. The goal is to prove that our aggressively quantized Hamming search maintains high semantic fidelity.

1.  **Recall Fidelity (Accuracy Tradeoffs)**:
    *   **Task**: Integrate a real-world benchmark dataset (e.g., **MS MARCO** or **SQuAD**) using the `HuggingFaceEmbedder`.
    *   **Metric**: Calculate **Recall@10** and **Recall@100** by comparing IsoSearch results against an exact Euclidean baseline.
    *   **Tuning**: Optimize Poincaré Ball scaling factors and LSH hyperplane counts to find the "Sweet Spot" between speed reduction and retrieval accuracy.

### Remaining Future Roadmap

1.  **System Partitioning (K-Means Refinement):** Scale the routing network to handle a much larger number of shards for multi-million document benchmarks.
2.  **Cache-Hitting Optimizations (Struct of Arrays):** Transition the object-oriented graph nodes in `HNSWGraph` into a tightly packed memory arena to align CPU L1 caching during traversal.
3.  **Production CLI/API**: Build a lightweight ingestion interface to allow external users to index and search their own document collections.
