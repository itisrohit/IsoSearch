//! Routing Network: Search space reduction.
//!
//! This module implements the first stage of the retrieval pipeline,
//! predicting which index partition (cluster) is most relevant to a query.

use crate::types::Embedding;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Trait defining the behavior of a partition router.
pub trait Router: Send + Sync {
    /// Given a query embedding, predict the index partition ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the routing operation fails (e.g. no partitions available).
    fn route(&self, query: &Embedding) -> Result<u64>;

    /// Get the total number of partitions managed by this router.
    fn partition_count(&self) -> usize;
}

/// K-Means based router for centroid-based partitioning.
#[derive(Debug, Serialize, Deserialize)]
pub struct KMeansRouter {
    centroids: Vec<Embedding>,
}

impl KMeansRouter {
    /// Create a new `KMeansRouter` from a set of centroids.
    #[must_use]
    pub const fn new(centroids: Vec<Embedding>) -> Self {
        Self { centroids }
    }
}

impl Router for KMeansRouter {
    /// Given a query embedding, predict the index partition ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the router has no centroids/partitions configured.
    fn route(&self, query: &Embedding) -> Result<u64> {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let diff = query - c;
                let dist_sq = diff.dot(&diff);
                (u64::try_from(i).unwrap_or(u64::MAX), dist_sq)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
            .ok_or_else(|| anyhow::anyhow!("Router has no partitions"))
    }

    fn partition_count(&self) -> usize {
        self.centroids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_empty_router_fails() {
        let router = KMeansRouter::new(vec![]);
        let query = Array1::from_elem(128, 0.0);
        let result = router.route(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_routing() {
        // Two centroids: one near 1.0, one near -1.0
        let dim = 4;
        let c1 = Array1::from_elem(dim, 1.0);
        let c2 = Array1::from_elem(dim, -1.0);
        let router = KMeansRouter::new(vec![c1, c2]);

        // Query near 1.0 should go to partition 0
        let q1 = Array1::from_elem(dim, 0.8);
        assert_eq!(router.route(&q1).unwrap(), 0);

        // Query near -1.0 should go to partition 1
        let q2 = Array1::from_elem(dim, -1.2);
        assert_eq!(router.route(&q2).unwrap(), 1);
    }
}
