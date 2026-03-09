//! Embedding Generation: API Integration
//!
//! This module implements the second stage of the retrieval pipeline,
//! communicating with external inference providers (like Groq) to generate
//! dense semantic representations of text.

use crate::types::Embedding;
use anyhow::{Result, anyhow};
use ndarray::Array1;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};

/// Interface for generating semantic embeddings.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Convert text into a dense vector representation.
    async fn embed(&self, text: &str) -> Result<Embedding>;

    /// Convert multiple texts in a batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
}

/// Request payload for OpenAI-compatible embeddings endpoints.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// Response payload for OpenAI-compatible embeddings endpoints.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Client for Groq's high-speed inference API.
#[derive(Clone, Debug)]
pub struct GroqEmbedder {
    client: Client,
    model: String,
    endpoint: String,
}

impl GroqEmbedder {
    /// Create a new `GroqEmbedder` client.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to build.
    pub fn new(api_key: &str, model: impl Into<String>) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| anyhow!("Invalid API key characters: {e}"))?;
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder().default_headers(headers).build()?;

        Ok(Self {
            client,
            model: model.into(),
            endpoint: "https://api.groq.com/openai/v1/embeddings".to_string(), // Groq OpenAI-compatible endpoint
        })
    }
}

#[async_trait::async_trait]
impl Embedder for GroqEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let mut batch = self.embed_batch(&[text]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow!("Groq returned an empty embedding array"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let req = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let res = self.client.post(&self.endpoint).json(&req).send().await?;

        if !res.status().is_success() {
            let err_text = res
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Groq embedding API failed: {err_text}"));
        }

        let parsed: EmbeddingResponse = res.json().await?;

        // Convert Vec<f32> into ndarray::Array1<f32>
        let result = parsed
            .data
            .into_iter()
            .map(|d| Array1::from_vec(d.embedding))
            .collect();

        Ok(result)
    }
}

/// Client for Hugging Face's FREE Inference API.
///
/// This uses the free Hugging Face inference endpoint which supports various
/// embedding models without requiring authentication (though authentication
/// provides better rate limits).
#[derive(Clone, Debug)]
pub struct HuggingFaceEmbedder {
    client: Client,
    #[allow(dead_code)] // Kept for potential debugging/logging use
    model: String,
    endpoint: String,
}

impl HuggingFaceEmbedder {
    /// Create a new `HuggingFaceEmbedder` client.
    ///
    /// # Arguments
    /// * `api_token` - Optional API token. Use empty string for unauthenticated requests.
    /// * `model` - The model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to build.
    pub fn new(api_token: &str, model: impl Into<String>) -> Result<Self> {
        let mut headers = header::HeaderMap::new();

        // Add authorization header if token is provided
        if !api_token.is_empty() && api_token != "your_token_here_optional" {
            let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {api_token}"))
                .map_err(|e| anyhow!("Invalid API token characters: {e}"))?;
            auth_value.set_sensitive(true);
            headers.insert(header::AUTHORIZATION, auth_value);
        }

        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let model_str = model.into();
        // Use HuggingFace router with hf-inference provider
        // Format: https://router.huggingface.co/hf-inference/models/{model_id}
        let endpoint = format!("https://router.huggingface.co/hf-inference/models/{model_str}");

        Ok(Self {
            client,
            model: model_str,
            endpoint,
        })
    }
}

#[async_trait::async_trait]
impl Embedder for HuggingFaceEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let mut batch = self.embed_batch(&[text]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow!("HuggingFace returned an empty embedding array"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        // HuggingFace API expects {"inputs": ["text1", "text2", ...]}
        let payload = serde_json::json!({
            "inputs": texts,
            "options": {
                "wait_for_model": true
            }
        });

        let res = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let err_text = res
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("HuggingFace API failed ({status}): {err_text}"));
        }

        // HuggingFace returns either:
        // - Single input: [[float, float, ...]]
        // - Batch input: [[[float, float, ...]], [[float, float, ...]]]
        let response_text = res.text().await?;

        // Try to parse as nested array (batch response)
        let embeddings: Vec<Vec<f32>> = serde_json::from_str(&response_text)
            .map_err(|e| anyhow!("Failed to parse HuggingFace response: {e}"))?;

        // Convert Vec<Vec<f32>> into Vec<Array1<f32>>
        let result = embeddings.into_iter().map(Array1::from_vec).collect();

        Ok(result)
    }
}
