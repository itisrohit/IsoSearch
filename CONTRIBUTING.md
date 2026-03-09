# Contributing to IsoSearch

Thank you for your interest in contributing to IsoSearch! This project is an experimental, performance-focused vector search engine, and we maintain high standards for code quality and computational efficiency.

## 🚀 Performance-First Principle
IsoSearch is an optimization research project. Every change to the core retrieval pipeline (hashing, graph traversal, projections) **must** be accompanied by benchmark data.

## 🛠 Quality & Standards
All contributions must pass the following automated checks before being merged:

1.  **Format**: We use `rustfmt`. Run `make fmt` to check.
2.  **Clippy**: We do not allow warnings. Run `make clippy`.
3.  **Tests**: All new logic must be accompanied by unit tests. Run `make test`.
4.  **Security**: We audit dependencies. Run `make audit`.

You can run the entire suite at once using:
```bash
make check
```

## 📋 Workflow
1.  **Fork the repository** and create your branch from `main`.
2.  **Implement your changes** and ensure they follow the [Architecture Specifications](docs/goal.md).
3.  **Benchmark your performance** if you touch the core engine.
4.  **Submit a Pull Request** using the template provided.

## ✍️ Commit Guidelines
We follow standard Conventional Commits:
*   `feat(scope): ...`
*   `fix(scope): ...`
*   `perf(scope): ...`
*   `docs(scope): ...`
*   `refactor(scope): ...`

---

*IsoSearch aims for O(1) retrieval. Keep the hot paths lean.*
