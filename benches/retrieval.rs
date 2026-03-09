#![allow(missing_docs, clippy::pedantic, clippy::nursery, clippy::unwrap_used)]
//! Retrieval benchmark suite for IsoSearch.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use isosearch::graph::{HNSWGraph, Node};
use isosearch::hashing::{BinaryQuantizer, LocalitySensitiveHasher, Quantizer, SimHasher};
use isosearch::indexing::BucketIndex;
use isosearch::normalization::{Normalizer, WhiteningNormalizer};
use isosearch::projection::{PoincareProjector, Projector, RandomProjector};
use ndarray::Array1;

/// Template for benchmarking the search pipeline.
pub fn bench_retrieval_latency(c: &mut Criterion) {
    let dim = 384;
    let target_dim = 128;

    // We mock a query
    let query_raw = Array1::from_elem(dim, 0.1);

    // Phase 1 Setup
    let corpus = vec![
        Array1::zeros(dim),
        Array1::ones(dim),
        Array1::from_elem(dim, 0.5),
        query_raw.clone(),
    ];
    let normalizer = WhiteningNormalizer::fit(&corpus).unwrap();
    let poincare = PoincareProjector::new();
    let random_proj = RandomProjector::new_gaussian(dim, target_dim);
    let hasher = SimHasher::new(target_dim, 64);
    let quantizer = BinaryQuantizer::new();

    let mut index = BucketIndex::new();
    let mut graph = HNSWGraph::new();

    let query_hashed = quantizer.quantize(
        &hasher.hash(&random_proj.project(&poincare.project(&normalizer.normalize(&query_raw)))),
    );

    // Seed intersection and graph logic
    for i in 0..10_000 {
        if let Some(&primary_hash) = query_hashed.first() {
            index.insert(primary_hash, i);
        }
        graph.nodes.insert(
            i,
            Node {
                id: i,
                hash: query_hashed.clone(),
                neighbors: vec![vec![]], // Stub for benchmarking
            },
        );
    }
    graph.entry_point = Some(0);

    // Benchmark 1: Math pipeline
    c.bench_function("normalize_and_project", |b| {
        b.iter(|| {
            let n = normalizer.normalize(black_box(&query_raw));
            let hp = poincare.project(black_box(&n));
            let _rp = random_proj.project(black_box(&hp));
        });
    });

    let n = normalizer.normalize(&query_raw);
    let hp = poincare.project(&n);
    let reduced = random_proj.project(&hp);

    // Benchmark 2: Hashing pipeline
    c.bench_function("hash_and_quantize", |b| {
        b.iter(|| {
            let h = hasher.hash(black_box(&reduced));
            let _q = quantizer.quantize(black_box(&h));
        });
    });

    // Benchmark 3: Index Lookup
    c.bench_function("bucket_intersection", |b| {
        b.iter(|| {
            let _candidates = index.intersect(black_box(&query_hashed));
        });
    });

    // Benchmark 4: HNSW Graph navigation
    c.bench_function("hnsw_traversal", |b| {
        b.iter(|| {
            let _res = graph.search(black_box(&query_hashed), black_box(10));
        });
    });

    // Baseline: General RAG (Naive Flat Exact Search)
    // Computes exact Euclidean distance over all 10,000 dense 384D vectors
    let mut flat_corpus = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        flat_corpus.push(Array1::from_elem(dim, 0.5)); // Dummy dense vector
    }

    c.bench_function("naive_flat_search", |b| {
        b.iter(|| {
            let mut best_dist = f32::MAX;
            for doc in &flat_corpus {
                let diff = black_box(&query_raw) - doc;
                let dist = diff.dot(&diff);
                if dist < best_dist {
                    best_dist = dist;
                }
            }
            black_box(best_dist);
        });
    });
}

criterion_group!(benches, bench_retrieval_latency);
criterion_main!(benches);
