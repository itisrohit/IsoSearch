//! HNSW (Hierarchical Navigable Small World) Graph Search
//!
//! This module implements Step 10: fast nearest neighbor search in Hamming space.
//! It uses a multi-layer graph to navigate towards the closest match efficiently.

use crate::types::ID;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for the document.
    pub id: ID,
    /// The binary hash (represented as words) for Hamming distance.
    pub hash: Vec<u64>,
    /// Neighbors at each level. `neighbors[0]` is the base layer.
    pub neighbors: Vec<Vec<ID>>,
}

/// The HNSW graph structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HNSWGraph {
    /// Map from document ID to its node data.
    pub nodes: HashMap<ID, Node>,
    /// The entry point ID for the search.
    pub entry_point: Option<ID>,
    /// Maximum level of the graph.
    pub max_level: usize,
}

impl HNSWGraph {
    /// Creates a new, empty `HNSWGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the Hamming distance between two hashed vectors (as u64 words).
    #[must_use]
    pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    /// Performs a greedy search at a specific level.
    #[must_use]
    pub fn search_level(&self, query_hash: &[u64], entry_id: ID, level: usize) -> ID {
        let mut curr_id = entry_id;
        let mut curr_dist = Self::hamming_distance(query_hash, &self.nodes[&curr_id].hash);
        let mut changed = true;

        while changed {
            changed = false;
            if let Some(node) = self
                .nodes
                .get(&curr_id)
                .filter(|n| level < n.neighbors.len())
            {
                for &neighbor_id in &node.neighbors[level] {
                    let dist = Self::hamming_distance(query_hash, &self.nodes[&neighbor_id].hash);
                    if dist < curr_dist {
                        curr_dist = dist;
                        curr_id = neighbor_id;
                        changed = true;
                    }
                }
            }
        }
        curr_id
    }

    /// Finds the nearest neighbors in Hamming space.
    ///
    /// # Arguments
    /// * `query_hash` - The quantized binary hash to search for.
    /// * `top_k` - Number of candidates to return.
    ///
    /// # Panics
    /// Panics if the internal priority queue state becomes inconsistent.
    #[must_use]
    pub fn search(&self, query_hash: &[u64], top_k: usize) -> Vec<(ID, u32)> {
        let Some(ep) = self.entry_point else {
            return Vec::new();
        };

        // 1. Navigate down the levels to the base layer
        let mut curr_ep = ep;
        for level in (1..=self.max_level).rev() {
            curr_ep = self.search_level(query_hash, curr_ep, level);
        }

        // 2. Perform a full search on the base layer (level 0)
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut to_visit = BinaryHeap::new();

        let dist = Self::hamming_distance(query_hash, &self.nodes[&curr_ep].hash);
        to_visit.push(SearchResult {
            id: curr_ep,
            distance: dist,
        });
        visited.insert(curr_ep);

        while let Some(SearchResult { id, distance }) = to_visit.pop() {
            candidates.push(SearchResult { id, distance });
            if candidates.len() > top_k {
                candidates.pop();
            }

            if let Some(node) = self.nodes.get(&id) {
                for &neighbor_id in &node.neighbors[0] {
                    if visited.insert(neighbor_id) {
                        let d = Self::hamming_distance(query_hash, &self.nodes[&neighbor_id].hash);
                        if candidates.len() < top_k || d < candidates.peek().unwrap().distance {
                            to_visit.push(SearchResult {
                                id: neighbor_id,
                                distance: d,
                            });
                        }
                    }
                }
            }
        }

        candidates
            .into_sorted_vec()
            .into_iter()
            .map(|res| (res.id, res.distance))
            .collect()
    }
}

#[derive(Eq, PartialEq)]
struct SearchResult {
    id: ID,
    distance: u32,
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Inverse for max-heap to act as min-heap if needed, or vice versa.
        // For to_visit, we want smallest distance first (min-heap).
        // For candidates, we want largest distance first (max-heap).
        // Since BinaryHeap is a max-heap, we use reverse for d.
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
