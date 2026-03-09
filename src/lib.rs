//! `isosearch` library core.

pub mod embedding;
/// Tools for geometric normalization and whitening.
pub mod normalization;
pub mod routing;

/// Common types used across the retrieval pipeline.
pub mod types {
    use ndarray::Array1;

    /// Representative of an embedding vector.
    pub type Embedding = Array1<f32>;

    /// Unique identifier for a document or partition.
    pub type ID = u64;
}
