//! Multi-Index Bucket Storage and Intersection
//!
//! This module implements Step 9 of the pipeline: Bucket Intersection.
//! It stores document IDs in hash-buckets to enable fast candidate retrieval
//! via set operations.

use crate::types::ID;
use std::collections::{HashMap, HashSet};

/// An inverted index for LSH buckets.
#[derive(Debug, Clone, Default)]
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
        let mut candidates = HashSet::new();
        for &h in hashes {
            if let Some(ids) = self.table.get(&h) {
                for &id in ids {
                    candidates.insert(id);
                }
            }
        }
        candidates.into_iter().collect()
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
