# IsoSearch

[![Rust 2024](https://img.shields.io/badge/Rust-2024-f34f29.svg?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Experimental](https://img.shields.io/badge/Status-Experimental-yellow.svg)](#)

> **⚠️ Experimental Status**: IsoSearch is an active research and development project. The architecture, pipelines, and benchmarks are currently experimental. It should not be used in critical production environments yet.

**IsoSearch** is a high-performance vector search engine designed to explore extreme optimizations in approximate nearest neighbor (ANN) retrieval. By mapping dense semantic embeddings into aggressively quantized bitspaces, IsoSearch aims to maximize retrieval speed and minimize memory footprint.

---

## Features

- **Quantized Hashing**: Projects 384D semantic embeddings into compact 64-bit binary fingerprints via Locality-Sensitive Hashing (SimHash).
- **Hamming Space Traversals**: Executes nearest-neighbor searches iteratively using highly optimized Bitwise XOR popcount operations.
- **Bucket-based Indexing**: Sidesteps linear indexing traps via constant-time $O(1)$ Hash Table intersections.
- **Non-Euclidean Reductions**: Leverages Hyperbolic Poincaré Ball projection to enforce structural compression of vector distributions.
- **CPU Native**: Optimized extensively for commodity hardware—no GPUs are strictly required.

---

## Prerequisites

The core engine requires a linear algebra backend (OpenBLAS/LAPACK) via `ndarray-linalg` to perform accelerated matrix operations.

**macOS:**
```bash
brew install openblas
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev gfortran
```

**Windows (via WSL2 — Recommended):**

1. Open **PowerShell as Administrator** and run:
```powershell
   wsl --install -d Ubuntu
```
   Reboot when prompted, then open the **Ubuntu** app to finish setup.

2. Inside WSL, run:
```bash
   cd ~
   sudo apt update && sudo apt install -y build-essential gfortran pkg-config libopenblas-dev libssl-dev
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
```
   To make sure path resolves correctly, set this:
```bash
   export OPENBLAS_DIR=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)
```
   You can persist it:
```bash
   echo 'export OPENBLAS_DIR=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)' >> ~/.bashrc
```

> **Note:** Always work from your Linux home directory (`cd ~`) inside WSL, not from `/mnt/c/...`. If `wsl --install` fails, ensure **Virtualization** is enabled in your BIOS (Task Manager → Performance → CPU → Virtualization: Enabled).

---

## Installation & Usage

1. **Clone the repository and install toolchains:**
```bash
   git clone https://github.com/itisrohit/IsoSearch.git
   cd IsoSearch
   make setup
```

2. **Build the optimized engine:**
```bash
   cargo build --release
```

3. **Run the simulation pipeline:**
```bash
   cargo run
```

---

## Performance & Benchmarks

IsoSearch is continuously benchmarked against standard "General RAG" flat Euclidean search architectures to validate its computational efficiency. 

* **Preliminary Micro-benchmarks (10,000 simulated documents)**: IsoSearch proved to be **~2.6x faster** than exact Euclidean distance scanning on consumer hardware (Apple M1), with a full pipeline latency of **~362.4 microseconds**. 
* **Recent Optimizations**:
  - **Zero-Allocation Rescoring**: Achieved a **292x speedup** in Euclidean distance checks (92ns → 315ps) by eliminating heap pressure through iterator-fusing.
  - **State-of-the-Art SIMD**: Implemented **Muła-Lemire** parallel lookup (arXiv:1611.07612) for sub-nanosecond Hamming searches (~478ps per 256-bit block).
  - **Parallel Scaling**: Achieved **3.3x scaling** via Rayon concurrency for batch projection and rescoring operations.
* **Deep Dive**: For full hardware profiling, pipeline mathematics, and theoretical big-O scalability implications, see exactly what we measured in the [Technical Analysis & Benchmarks](docs/technical_analysis.md) report.

To run the benchmarks locally:
```bash
make bench
```

---

## Quality & Development

The project maintains rigorous code quality and security standards through automated enforcement.

* **Git Hooks**: Managed by Lefthook to automatically run formatting, testing, and lint checks.
* **Security & License Monitoring**: Automated via `cargo-audit` and `cargo-deny`.

To manually run the comprehensive quality suite:
```bash
make check
```

---

## Technical Docs

Detailed technical objectives and stage-by-stage architectural plans are strictly documented in [docs/goal.md](docs/goal.md).
