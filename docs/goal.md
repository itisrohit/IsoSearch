# Technical Specification: IsoSearch Retrieval System

## Objectives and Performance Targets

The primary objective is the development of a **sub-millisecond approximate nearest neighbor (ANN) retriever** capable of indexing and searching millions of documents with minimal computational overhead.

| Metric | Target Specification |
| :--- | :--- |
| **Search Latency** | < 3–5 ms per query |
| **Dataset Scale** | 1M to 50M documents |
| **Index Memory Footprint** | < 1 GB total index size |
| **Retrieval Accuracy** | Recall@10 ≥ 90% (relative to brute-force exact search) |
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
7. **[ ] Locality-Sensitive Hashing (LSH)**
8. **[ ] Binary Quantization**
9. **[ ] Bucket Intersection**
10. **[ ] HNSW Graph Traversal** (Hamming Space Search)
11. **[ ] Precision Rescoring** (Full Vector Reranking)

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

| Component | Implementation Strategy |
| :--- | :--- |
| **Core Engine** | Pure Rust with SIMD (AVX-512/NEON) intrinsics for vector math |
| **Linear Algebra** | `ndarray` with `ndarray-linalg` (OpenBLAS/LAPACK backend) |
| **Graph Indexing** | Native Rust HNSW implementation (optimizing for cache locality) |
| **Inference Engine** | API-based client (`reqwest`, `tokio`) connecting to HuggingFace Inference API |
| **Concurrency** | `Rayon` for parallel index building and search space traversal |
| **Transformation Kernels** | Optimized Rust SIMD kernels for Whitening and Poincaré mapping |
| **Serialization** | `serde` and `bincode` for high-speed, zero-copy index persistence |

---

## Evaluation Framework (Proposed Experiments)

1.  **Geometric Normalization Impact**: Benchmark Recall@10 with and without whitening and mean-centering.
2.  **Manifold Fidelity**: Compare Euclidean vs. Poincaré-ball projections across datasets with varying hierarchical structures.
3.  **Dimensionality Sensitivity**: Quantify the accuracy-latency Pareto curve for 256, 128, and 64-dimensional projections.
4.  **Bit-depth Optimization**: Evaluate recall performance across 64, 128, and 256-bit LSH configurations.
5.  **Routing Precision**: Quantify recall loss introduced by index partitioning vs. global search operations.
