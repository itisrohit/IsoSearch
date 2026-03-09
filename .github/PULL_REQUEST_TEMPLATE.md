## Description
<!-- Provide a clear and concise description of the changes in this PR. -->

## Type of Change
- [ ] 🚀 Performance Optimization (SIMD, Concurrency, etc.)
- [ ] ✨ New Feature
- [ ] 🐛 Bug Fix
- [ ] 📝 Documentation Update
- [ ] 🔧 Refactoring

## Performance Benchmarks (Mandatory for core changes)
### **Hardware / Environment**
- **OS**: [e.g. macOS, Linux, Windows]
- **CPU**: [e.g. M1 Silicon, Intel i7-12700K]
- **Arch**: [e.g. ARM64, x86_64]

### **Comparison**
<!-- 
If this PR affects retrieval performance, please provide Criterion benchmark results.
Example: 
- Before: 938.6 µs
- After: 370.1 µs (~2.5x speedup)
-->

## Testing & Validation
- [ ] `make check` passes locally (Formatting, Clippy, Tests)
- [ ] I have added new **Unit Tests** to verify the logic
- [ ] I have added/updated **Benchmarks** (under `benches/`) to measure performance impact
- [ ] I have verified the changes on the hardware specified above

## Checklist
- [ ] My code follows the Rust standard formatting (`rustfmt`)
- [ ] I have ran `clippy` and it has no warnings (`-D warnings`)
- [ ] I have updated the documentation (`README.md`, `docs/*.md`) where necessary
- [ ] Commit messages follow the Conventional Commit pattern (`feat:`, `fix:`, `perf:`, etc.)
