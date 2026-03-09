.PHONY: check test fmt clippy audit bench clean

# Default task: Run all common quality checks
check: fmt clippy test audit

# Run tests
test:
	cargo test --all-features

# Check formatting
fmt:
	cargo fmt --all -- --check

# Run clippy with strict warnings
clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Security audit of dependencies
audit:
	cargo audit || echo "Please install cargo-audit: cargo install cargo-audit"

# Run benchmarks
bench:
	cargo bench

# Clean build artifacts
clean:
	cargo clean

# Install recommended tools
setup:
	rustup component add clippy rustfmt
	cargo install cargo-audit cargo-watch bacon
