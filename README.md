# IsoSearch

IsoSearch is a sub-millisecond approximate nearest neighbor (ANN) retriever designed for large-scale document search (1M to 50M documents). It is optimized for minimal memory consumption, targeting an index footprint of less than 1 GB.

## Requirements

The core engine is implemented in Rust (2024 Edition) and requires a linear algebra backend.

### System Dependencies

*   **macOS**: `brew install openblas`
*   **Linux (Ubuntu/Debian)**: `sudo apt-get install libopenblas-dev gfortran`

## Installation and Setup

1.  Initialize the development environment:
    ```bash
    make setup
    ```
2.  Build the optimized release binary:
    ```bash
    cargo build --release
    ```

## Quality Assurance

The project maintains high code quality and security standards through automated checks.

### Quality Protection

*   **Git Hooks**: Managed by Lefthook. Automatically runs formatting, linting, and tests on every commit to prevent poor-quality code from entering the history.
*   **Continuous Integration**: Automated gates on every PR using GitHub Actions.
*   **Security**: Automated dependency auditing via `cargo-audit`.
*   **Compliance**: License monitoring through `cargo-deny`.

### Local Development Commands

*   Run comprehensive quality suite: `make check`
*   Execute unit and integration tests: `make test`
*   Perform micro-benchmarking: `make bench`

## Specifications

Detailed technical objectives and architectural details are documented in [docs/goal.md](docs/goal.md).
