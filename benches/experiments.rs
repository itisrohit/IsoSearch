#![allow(missing_docs, clippy::pedantic, clippy::nursery, clippy::unwrap_used)]
//! Evaluation framework benchmarks covering the pipeline optimizations
//! proposed in docs/goal.md.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use isosearch::graph::HNSWGraph;
use isosearch::hashing::{BinaryQuantizer, LocalitySensitiveHasher, Quantizer, SimHasher};
use isosearch::projection::{Projector, RandomProjector};
use ndarray::Array1;

/// Experiment 3: Dimensionality Sensitivity
/// Benchmarks the latency cost of projecting from 384D to varying low-dim spaces.
fn bench_dimensionality_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensionality_projection");
    let input_dim = 384;
    let query = Array1::from_elem(input_dim, 0.5);

    let dims = [64, 128, 256];
    for &target_dim in &dims {
        let projector = RandomProjector::new_gaussian(input_dim, target_dim);
        group.bench_function(format!("384D_to_{target_dim}D"), |b| {
            b.iter(|| {
                let _rp = projector.project(black_box(&query));
            });
        });
    }
    group.finish();
}

/// Experiment 4: Bit-depth Optimization
/// Benchmarks the latency of hashing and quantizing varying bit depths.
fn bench_bit_depth_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_hashing_and_quantization");
    let input_dim = 128; // Standard intermediate dim
    let query = Array1::from_elem(input_dim, 0.5);

    let quantizer = BinaryQuantizer::new();

    let bit_depths = [64, 128, 256];
    for &bits in &bit_depths {
        let hasher = SimHasher::new(input_dim, bits);
        group.bench_function(format!("{bits}bit_hash"), |b| {
            b.iter(|| {
                let h = hasher.hash(black_box(&query));
                let _q = quantizer.quantize(black_box(&h));
            });
        });
    }
    group.finish();
}

/// Compare computing Exact Euclidean vs Hamming Distance (Baseline vs SIMD)
fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");
    let dim = 384;
    let vec_a = Array1::from_elem(dim, 0.1);
    let vec_b = Array1::from_elem(dim, 0.9);

    group.bench_function("bench_euclidean_allocating_baseline", |b| {
        b.iter(|| {
            // This represents the old, slow way that allocated a new vector
            let diff = black_box(&vec_a) - black_box(&vec_b);
            let _dist = diff.dot(&diff);
        });
    });

    group.bench_function("bench_euclidean_zero_allocation", |b| {
        b.iter(|| {
            // This represents the new, fast, zero-allocation way
            let _dist: f32 = vec_a
                .iter()
                .zip(vec_b.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
        });
    });

    // 256 bits = 4 u64 words
    let hash_a = vec![
        0x1234_5678_90AB_CDEF,
        0x1234_5678_90AB_CDEF,
        0x1234_5678_90AB_CDEF,
        0x1234_5678_90AB_CDEF,
    ];
    let hash_b = vec![
        0xFEDC_BA09_8765_4321,
        0xFEDC_BA09_8765_4321,
        0xFEDC_BA09_8765_4321,
        0xFEDC_BA09_8765_4321,
    ];

    group.bench_function("baseline_hamming_256bit", |b| {
        b.iter(|| {
            let _dist = HNSWGraph::hamming_distance(black_box(&hash_a), black_box(&hash_b));
        });
    });

    group.bench_function("simd_fast_hamming_256bit", |b| {
        b.iter(|| {
            let _dist = HNSWGraph::fast_hamming_distance(black_box(&hash_a), black_box(&hash_b));
        });
    });

    group.finish();
}

/// Experiment 6: Concurrency Speedup
/// Benchmarks parallel search and projection using Rayon.
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency_parallelism");
    let dim = 384;
    let target_dim = 128;
    let n_queries = 100;

    let queries = vec![Array1::from_elem(dim, 0.5); n_queries];
    let projector = RandomProjector::new_gaussian(dim, target_dim);

    group.bench_function("sequential_projection_100x", |b| {
        b.iter(|| {
            let _res: Vec<_> = queries.iter().map(|q| projector.project(q)).collect();
        });
    });

    group.bench_function("parallel_projection_100x", |b| {
        b.iter(|| {
            let _res = projector.par_project(black_box(&queries));
        });
    });

    group.finish();
}

/// Experiment 7: Serialization Latency
/// Benchmarks bincode save/load for a small graph.
fn bench_serialization_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization_persistence");
    let mut graph = HNSWGraph::new();
    let id = 1;
    graph.nodes.insert(
        id,
        isosearch::graph::Node {
            id,
            hash: vec![0x1234, 0x5678],
            neighbors: vec![vec![2, 3, 4]],
        },
    );
    graph.entry_point = Some(id);

    let path = "bench_graph.bin";

    group.bench_function("bincode_save_graph", |b| {
        b.iter(|| {
            graph.save(black_box(path)).unwrap();
        });
    });

    group.bench_function("bincode_load_graph", |b| {
        b.iter(|| {
            let _g = HNSWGraph::load(black_box(path)).unwrap();
        });
    });

    std::fs::remove_file(path).ok();
    group.finish();
}

criterion_group!(
    experiments,
    bench_dimensionality_sensitivity,
    bench_bit_depth_optimization,
    bench_distance_computation,
    bench_parallel_operations,
    bench_serialization_latency
);
criterion_main!(experiments);
