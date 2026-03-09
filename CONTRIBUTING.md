# Contributing to IsoSearch

Thank you for your interest in contributing to IsoSearch! This project is an experimental, performance-focused vector search engine, and we maintain high standards for code quality and computational efficiency.

## Types of Contributions

We welcome contributions across several domains. Whether you are a systems engineer, a mathematical researcher, or a documentation expert, there is a place for your expertise:

### 1. Core Retrieval Engine
*   **SIMD Optimization**: Enhancing existing NEON/AVX2 kernels or adding support for new architectures (e.g., SVE, AVX-512).
*   **Memory Architecture**: Improving cache locality through Struct-of-Arrays (SoA) layouts or memory arena allocation.
*   **Graph Logic**: Refining HNSW traversal algorithms to reduce the number of candidate comparisons.

### 2. Research & Benchmarking (Co-Authorship Opportunity)
As an experimental project, we are looking for contributors to help validate our mathematical hypotheses.
*   **Dataset Benchmarks**: Run retrieval performance tests on datasets like MNIST (384D) or Sift1M.
*   **Metrics Analysis**: Develop scripts for p99 latency percentiles or `malloc` footprint analysis.
*   **Scientific Recognition**: Significant contributors to Phase 5 (Recall Evaluation) will be invited as **co-authors** on any future technical papers documenting these results.

### 3. Infrastructure & Tooling
*   **CLI Development**: Building a robust command-line interface for indexing and searching local document sets.
*   **Dataset Loaders**: Creating efficient parsers for standard vector formats (HDF5, Numpy, etc.).
*   **CI/CD**: Improving our automated testing and benchmarking pipelines.

### 4. Documentation & Developer Experience
*   **Technical Writing**: Clarifying complex mathematical concepts in `docs/`.
*   **Examples & Tutorials**: Building "Getting Started" guides for new users.
*   **Bug Reports**: Identifying edge cases or performance regressions.

## Workflow

1.  **Find an Issue**: Browse the GitHub Issues/Discussions with `help wanted` or `research` labels.
2.  **Fork the Repository** and create a feature branch.
3.  **Implement & Verify**: Ensure your code passes all local checks (`make check`).
4.  **Benchmark (Mandatory for Core)**: If touching hot-paths, include Criterion benchmark output in your PR.
5.  **Submit a Pull Request**: Use our standardized template to document your changes and environment.

## Commit Guidelines
We follow standard Conventional Commits:
*   `feat(scope)`: New features or capabilities.
*   `fix(scope)`: Bug fixes.
*   `perf(scope)`: Performance-only optimizations.
*   `docs(scope)`: Documentation improvements.
*   `refactor(scope)`: Code changes that neither fix a bug nor add a feature.
*   `bench(scope)`: New benchmarks or research results.

---

*IsoSearch aims for O(1) retrieval. Keep the hot paths lean.*
