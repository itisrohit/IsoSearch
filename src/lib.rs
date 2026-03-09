//! `isosearch` library core.

pub mod embedding;
/// Locality-Sensitive Hashing (LSH) and binary quantization.
pub mod hashing;
/// Bucket-based candidate indexing and intersection.
pub mod indexing;
/// Tools for geometric normalization and whitening.
pub mod normalization;
/// Space projection transformations (Hyperbolic, Random).
pub mod projection;
pub mod routing;

/// Common types used across the retrieval pipeline.
pub mod types {
    use ndarray::Array1;

    /// Representative of an embedding vector.
    pub type Embedding = Array1<f32>;

    /// Unique identifier for a document or partition.
    pub type ID = u64;
}
