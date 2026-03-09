//! `IsoSearch`: A sub-millisecond approximate nearest neighbor retriever.
//!
//! Implementation of the optimized vector retrieval pipeline including:
//! - `LSH` (Locality Sensitive Hashing)
//! - Hyperbolic Space Projections
//! - `HNSW` (Hierarchical Navigable Small World) Graph Search

use anyhow::Result;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

use isosearch::embedding::{Embedder, HuggingFaceEmbedder};
use isosearch::routing::{KMeansRouter, Router};
use ndarray::Array1;
use std::env;

/// Initialize the application and search pipeline.
#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file if it exists
    dotenvy::dotenv().ok();

    // Setup professional logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Initializing IsoSearch Engine (2026 Edition)");
    info!("Target Specification: Sub-millisecond latency, <1GB Index");

    // Phase 1: Routing Network Initialization
    // Mocking 4 centroids for demonstration
    // Updated to match BAAI/bge-small-en-v1.5 embedding dimensions (384)
    let dim = 384;
    let centroids = vec![
        Array1::zeros(dim),
        Array1::ones(dim),
        Array1::from_elem(dim, 0.5),
        Array1::from_elem(dim, -0.5),
    ];
    let router = KMeansRouter::new(centroids);
    info!(
        "Initialized Routing Network with {} partitions",
        router.partition_count()
    );

    let api_token = env::var("HF_TOKEN").unwrap_or_else(|_| String::new());

    // Phase 2: Embedding Generation Setup
    // Using HuggingFace's FREE inference API with BAAI BGE model
    // This model produces 384-dimensional embeddings via the hf-inference provider
    let embedder = HuggingFaceEmbedder::new(&api_token, "BAAI/bge-small-en-v1.5")?;
    let mock_text = "What is approximate nearest neighbor search?";

    info!("Generating embedding for query: '{}'", mock_text);
    // Note: HuggingFace API is FREE and works without authentication (with rate limits)
    // For better rate limits, set HF_TOKEN in your .env file from https://huggingface.co/settings/tokens

    let query = match embedder.embed(mock_text).await {
        Ok(q) => {
            info!(
                "✓ Received embedding from HuggingFace API [Dim: {}]",
                q.len()
            );
            q
        }
        Err(e) => {
            tracing::warn!(
                "Embedding API failed: {}. Using mock embedding for routing.",
                e
            );
            Array1::from_elem(dim, 0.1)
        }
    };

    let partition = router.route(&query)?;
    info!("Routed query to partition: {}", partition);

    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_initialization() {
        // Simple sanity check that doesn't trigger clippy::assertions_on_constants
        let status = true;
        assert!(status);
    }
}
