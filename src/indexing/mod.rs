//! Multi-Index Bucket Storage and Intersection
//!
//! This module implements Step 9 of the pipeline: Bucket Intersection.
//! It stores document IDs in hash-buckets to enable fast candidate retrieval
//! via set operations.

use crate::types::ID;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// An inverted index for LSH buckets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BucketIndex {
    /// Maps a 64-bit quantized hash to a list of document IDs.
    pub table: HashMap<u64, Vec<ID>>,
}

impl BucketIndex {
    /// Creates a new, empty `BucketIndex`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a document ID into a specific hash bucket.
    pub fn insert(&mut self, hash: u64, id: ID) {
        self.table.entry(hash).or_default().push(id);
    }

    /// Retrieves all document IDs that reside in the specified buckets.
    ///
    /// This performs an "OR" intersection (union of buckets) to maximize recall
    /// before precision reranking.
    #[must_use]
    pub fn intersect(&self, hashes: &[u64]) -> Vec<ID> {
        if hashes.is_empty() {
            return Vec::new();
        }

        // Fast-path: Single hash lookup
        if hashes.len() == 1 {
            return self.table.get(&hashes[0]).cloned().unwrap_or_default();
        }

        // Optimized Multi-hash intersection (Union)
        // Memory-aligned collect then sort/dedup is faster than HashSet for these sizes.
        let mut candidates = Vec::new();
        for &h in hashes {
            if let Some(ids) = self.table.get(&h) {
                candidates.extend_from_slice(ids);
            }
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Serialization: Save the index to a file.
    ///
    /// # Errors
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let encoded: Vec<u8> = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Serialization: Load the index from a file.
    ///
    /// # Errors
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let decoded: Self = bincode::deserialize(&buffer)?;
        Ok(decoded)
    }
}

/// A primary storage for full-precision embeddings used for rescoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorStore {
    /// Maps document ID to its full embedding vector.
    pub storage: HashMap<ID, crate::types::Embedding>,
}

impl VectorStore {
    /// Creates a new, empty `VectorStore`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an embedding into the store.
    pub fn insert(&mut self, id: ID, embedding: crate::types::Embedding) {
        self.storage.insert(id, embedding);
    }

    /// Calculates exact Euclidean distances for a set of candidates.
    /// Parallelized using Rayon for massive performance gains during rescoring.
    ///
    /// # Returns
    /// A sorted list of (ID, distance) from nearest to farthest.
    #[must_use]
    pub fn rescore(&self, query: &crate::types::Embedding, candidates: &[ID]) -> Vec<(ID, f32)> {
        let mut scored: Vec<(ID, f32)> = candidates
            .par_iter()
            .filter_map(|id| {
                self.storage.get(id).map(|vec| {
                    let diff = query - vec;
                    let dist_sq = diff.dot(&diff);
                    (*id, dist_sq)
                })
            })
            .collect();

        scored.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_intersection() {
        let mut index = BucketIndex::new();

        // Insert dummy documents
        index.insert(0xABC, 1);
        index.insert(0xABC, 2);
        index.insert(0xDEF, 3);

        // Query for 0xABC
        let res = index.intersect(&[0xABC]);
        assert_eq!(res.len(), 2);
        assert!(res.contains(&1));
        assert!(res.contains(&2));

        // Query for intersection (union) of 0xABC and 0xDEF
        let res = index.intersect(&[0xABC, 0xDEF]);
        assert_eq!(res.len(), 3);
    }
}
