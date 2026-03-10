# Benchmarking V2: The Research Framework

This document outlines the Phase 5 research strategy for **IsoSearch**. Following the initial systems optimization (SIMD, Rayon, Bincode), we are transitioning from "building for speed" to "measuring for truth." This framework is designed to isolate the mathematical advantages of the IsoSearch pipeline and identify the true "Break-even Point" for performance vs. accuracy.

---

## 1. The Rescoring Tax Analysis

**Hypothesis**: "Rescoring top-K candidates provides a significant recall boost with negligible (<1µs) overhead for $K \le 1000$ due to zero-allocation iterator fusing."

### Experiment Design:
*   **Corpus**: 10,000 document vector set.
*   **Queries**: 100 semantic queries via `HuggingFaceEmbedder`.
*   **Variable**: Rescore depth level ($K$ = 10, 25, 50, 100, 200, 500, 1000).
*   **Metrics**:
    *   **Hamming search latency**: Isolated traversal cost.
    *   **Rescoring latency**: Vectorized dot-product cost for $K$ candidates.
    *   **Total E2E latency**: Direct comparison to exact flat scan.
    *   **Recall@K**: Comparison against absolute "Ground Truth" Euclidean search.

### Goal: Identify the "Pareto Frontier"
We must determine the optimal $K$ where we maximize recall without sacrificing our sub-millisecond target. Currently, zero-allocation rescoring allows us to consider much larger candidate sets than previously feasible.

---

## 2. Poincaré vs. Euclidean: A/B Testing

**Question**: Does the Poincaré Ball projection actually improve retrieval accuracy for semantic embeddings, or is it purely computational overhead?

### Dual-Pipeline Comparison:
| Phase | Pipeline A: Euclidean | Pipeline B: Poincaré |
| :--- | :--- | :--- |
| **Projection** | None / LSH Only | Poincaré + LSH |
| **Search** | Hamming Space | Hamming Space |
| **Rescore** | Euclidean | Euclidean |

### Decision Rule:
If `(recall_gain / latency_cost) < 0.1%`, the Poincaré projection will be moved to an optional/config-only feature to prevent "Vanity Latency."

---

## 3. Advanced Big-O & Production Metrics

### A. Latency Percentiles (p99)
Average speed is deceptive. We must measure the "Tail Latency" to ensure the HNSW graph search doesn't get caught in local minima for specific complex queries.
*   **Target**: p50, p95, and **p99** measurements over 10,000 iterations.

### B. Memory Footprint Analysis
The core value proposition of IsoSearch is its ability to fit billion-scale indices into standard server memory.
*   **FAISS/HNSW Baseline**: ~1.5 KB per vector (384D f32 + graph).
*   **IsoSearch Target**: ~100 Bytes per vector (64-bit hash + graph).
*   **Goal**: Demonstrate a **~90% memory reduction** without sacrificing standard RAG recall.

### C. Recall-vs-Bitrate (Compression Tuning)
Test accuracy scaling across different quantization widths:
*   **32-bit**: Half u64 word.
*   **64-bit**: Single u64 word (Current).
*   **128-bit**: Two words (SIMD-optimized NEON/AVX2).
*   **256-bit**: Four words.

---

## 4. Research Positioning

IsoSearch is not just another LSH implementation. Our unique contribution lies in the **Geometric Quantization** of hierarchical data.

### The Novel Claims:
1.  **Hierarchy Preservation**: Poincaré projection preserves the latent tree-like structure of semantic embeddings better than linear quantization.
2.  **Hardware-Native Recovery**: SIMD-optimized Hamming distance provides "Exact Recovery" of candidates at speeds that beat GPU-based code for commodity server deployments.
3.  **Principled Rescoring**: We provide a deterministic way to guarantee Recall@100 while maintaining a **2.6x** speed advantage over FAISS-less flat systems.

---

## Research Roadmap

### Rescoring Tax Analysis
*   Implement break-even benchmark script.
*   Identify optimal $K$ for rescoring depth.

### Poincaré A/B Test
*   Isolate the latency of the hyperbolic stage.
*   Validate recall gain on clustered/hierarchical datasets.

### Big-O Scaling
*   p99 latency analysis under load.
*   Measure actual `malloc` footprints for million-vector sets.

### Multi-Domain Validation
*   Test across 3 distinct datasets: Text (MS MARCO), Image (Sift1M), and Code (GitHub).
*   Publish final technical whitepaper.

---

*Is the math worth the latency? We will find out through empirical proof.*
