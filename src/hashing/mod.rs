//! Locality-Sensitive Hashing (LSH)
//!
//! This module implements hashing techniques that map similar vectors to the same
//! or nearby hash buckets with high probability. This is critical for sub-millisecond
//! retrieval in massive datasets.

use crate::types::Embedding;
use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

/// Type alias for a binary hash fingerprint.
pub type Hash = BitVec<u64, Lsb0>;

/// Interface for locality-sensitive hashing algorithms.
pub trait LocalitySensitiveHasher {
    /// Generates a binary hash for a given embedding.
    ///
    /// This method is marked with `#[inline]` to enable cross-crate
    /// optimizations and eliminate function call overhead in hot paths.
    fn hash(&self, vector: &Embedding) -> Hash;
}

/// Sign-Random-Projection (`SimHash`) implementation for Angular Similarity.
///
/// It projects vectors onto random hyperplanes and takes the sign of the result.
/// This preserves the cosine similarity between original vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimHasher {
    /// Random hyperplanes for projection (K x Dim matrix).
    /// Each row represents a hyperplane.
    pub hyperplanes: ndarray::Array2<f32>,
}

impl SimHasher {
    /// Creates a new `SimHasher` with random Gaussian hyperplanes.
    ///
    /// # Arguments
    /// * `dim` - Dimensionality of the input vectors.
    /// * `bits` - Number of bits in the resulting hash (number of hyperplanes).
    ///
    /// # Panics
    /// Panics if the Gaussian distribution initialization fails.
    #[must_use]
    pub fn new(dim: usize, bits: usize) -> Self {
        use rand::distributions::Distribution;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        let hyperplanes = ndarray::Array2::from_shape_fn((bits, dim), |_| dist.sample(&mut rng));

        Self { hyperplanes }
    }
}

impl LocalitySensitiveHasher for SimHasher {
    #[inline]
    fn hash(&self, vector: &Embedding) -> Hash {
        let projections = self.hyperplanes.dot(vector);
        let mut bits = BitVec::with_capacity(projections.len());

        for &val in &projections {
            bits.push(val >= 0.0);
        }

        bits
    }
}

/// Interface for vector quantization.
pub trait Quantizer {
    /// Compresses a high-dimensional hash into a compact binary representation.
    ///
    /// This method is marked with `#[inline]` to enable cross-crate
    /// optimizations and eliminate function call overhead in hot paths.
    fn quantize(&self, hash: &Hash) -> Vec<u64>;
}

/// A quantizer that packs bits into 64-bit unsigned integers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinaryQuantizer;

impl BinaryQuantizer {
    /// Creates a new `BinaryQuantizer`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Quantizer for BinaryQuantizer {
    #[inline]
    fn quantize(&self, hash: &Hash) -> Vec<u64> {
        hash.as_raw_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_simhash_length() {
        let dim = 128;
        let bits = 64;
        let hasher = SimHasher::new(dim, bits);

        let v = Array1::from_elem(dim, 0.5);
        let hash = hasher.hash(&v);

        assert_eq!(hash.len(), bits);
    }

    #[test]
    fn test_simhash_similarity_preservation() {
        let dim = 128;
        let bits = 256; // More bits for better statistical resolution
        let hasher = SimHasher::new(dim, bits);

        // v1 and v2 are very similar (positive correlation)
        let v1 = Array1::from_elem(dim, 1.0);
        let v2 = Array1::from_elem(dim, 0.9);

        // v3 is the opposite of v1 (negative correlation)
        let v3 = Array1::from_elem(dim, -1.0);

        let h1 = hasher.hash(&v1);
        let h2 = hasher.hash(&v2);
        let h3 = hasher.hash(&v3);

        // Hamming distance between similar vectors should be small
        let dist_12 = (h1.clone() ^ h2).count_ones();
        // Hamming distance between opposite vectors should be large
        let dist_13 = (h1 ^ h3).count_ones();

        assert!(dist_12 < dist_13);
        assert!(dist_12 < bits / 4); // Roughly < 25% difference
        assert!(dist_13 > bits * 3 / 4); // Roughly > 75% difference
    }
}
