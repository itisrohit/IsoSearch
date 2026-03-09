//! Geometric Normalization & Whitening
//!
//! This module addresses anisotropy in embedding spaces by centering the dataset
//! and decorrelating the vector dimensions using PCA-based whitening.

use crate::types::Embedding;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Interface for geometric vector normalization.
pub trait Normalizer {
    /// Applies the transformation to the input vector.
    fn normalize(&self, vector: &Embedding) -> Embedding;
}

/// A normalizer that centers embeddings around the mean and applies a
/// whitening transformation to ensure unit covariance across the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteningNormalizer {
    /// The mean vector of the training dataset.
    pub mean: Array1<f32>,
    /// The transformation matrix mapping raw embeddings into
    /// a decorrelated space.
    pub transformation: Array2<f32>,
}

impl WhiteningNormalizer {
    /// Fits a whitening transformation to the provided corpus using SVD.
    ///
    /// # Arguments
    /// * `embeddings` - A collection of vectors characterizing the target manifold.
    ///
    /// # Errors
    /// Returns an error if the dataset is empty or if SVD fails.
    pub fn fit(embeddings: &[Embedding]) -> Result<Self> {
        use ndarray_linalg::SVD as SVDTrait;

        if embeddings.is_empty() {
            return Err(anyhow!("Cannot fit normalizer to an empty dataset."));
        }

        let n_samples = embeddings.len();
        let dim = embeddings[0].len();

        // 1. Compute Mean (Parallelized with Rayon)
        let mean_sum = embeddings.par_iter().cloned().reduce(
            || Array1::zeros(dim),
            |mut a, b| {
                a += &b;
                a
            },
        );

        #[allow(clippy::cast_precision_loss)]
        let n_samples_f32 = n_samples as f32;
        let mean = mean_sum / n_samples_f32;

        // 2. Center Data & Construct Matrix (Parallelized with Rayon)
        let mut data_matrix = Array2::<f32>::zeros((n_samples, dim));
        data_matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                row.assign(&(&embeddings[i] - &mean));
            });

        let (_, s, vt_opt) = SVDTrait::svd(&data_matrix, false, true)
            .map_err(|e| anyhow!("SVD computation failed: {e}"))?;

        let vt = vt_opt.ok_or_else(|| anyhow!("SVD did not return V^T matrix"))?;
        let s = s; // Singular values (dim,)

        // 4. Compute Whitening Matrix
        // Scale singular values to match covariance eigenvalues: variance = s^2 / (n-1)
        // So scale = 1 / sqrt(s^2 / (n-1)) = sqrt(n-1) / s
        #[allow(clippy::cast_precision_loss)]
        let n_minus_1_sqrt = ((n_samples - 1) as f32).sqrt();
        let epsilon = 1e-10;

        let mut scales = Array1::<f32>::zeros(dim);
        for i in 0..std::cmp::min(dim, s.len()) {
            scales[i] = n_minus_1_sqrt / (s[i] + epsilon);
        }

        // Transformation = Vt.T * diag(scales)
        // This maps input vectors into the principal component space and normalizes variance.
        let diag_scales = Array2::from_diag(&scales);
        let transformation = vt.t().dot(&diag_scales);

        Ok(Self {
            mean,
            transformation,
        })
    }

    /// Parallelized batch normalization for efficient Phase 4 processing.
    #[must_use]
    pub fn par_normalize(&self, vectors: &[Embedding]) -> Vec<Embedding> {
        vectors.par_iter().map(|v| self.normalize(v)).collect()
    }

    /// Serialization: Save the normalizer to a file.
    ///
    /// # Errors
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let encoded: Vec<u8> = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Serialization: Load the normalizer from a file.
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

impl Normalizer for WhiteningNormalizer {
    fn normalize(&self, vector: &Embedding) -> Embedding {
        let centered = vector - &self.mean;
        // Transform: y = T^T * centered
        self.transformation.t().dot(&centered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_whitening_centering() {
        // Create a highly biased dataset
        let embeddings = vec![
            Array1::from_vec(vec![10.0, 20.0]),
            Array1::from_vec(vec![11.0, 21.0]),
            Array1::from_vec(vec![12.0, 22.0]),
        ];

        let normalizer = WhiteningNormalizer::fit(&embeddings).expect("Fit failed");

        // After whitening, the mean of transformed vectors should be zero
        let mut sum = Array1::zeros(2);
        for e in &embeddings {
            sum += &normalizer.normalize(e);
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = sum / (embeddings.len() as f32);

        // Tolerance for floating point noise
        assert!(avg.mapv(|x: f32| x.abs()).sum() < 1e-4);
    }

    #[test]
    fn test_whitening_decorrelation() {
        // Highly correlated inputs: y = 2x + noise
        let embeddings = vec![
            Array1::from_vec(vec![1.0, 2.1]),
            Array1::from_vec(vec![2.0, 3.9]),
            Array1::from_vec(vec![3.0, 6.2]),
            Array1::from_vec(vec![4.0, 7.8]),
            Array1::from_vec(vec![5.0, 10.1]),
        ];

        let normalizer = WhiteningNormalizer::fit(&embeddings).expect("Fit failed");

        // Transform the dataset
        let whitened: Vec<Embedding> = embeddings.iter().map(|e| normalizer.normalize(e)).collect();

        // Compute covariance of whitened data
        // It should be close to the Identity Matrix
        #[allow(clippy::cast_precision_loss)]
        let n = whitened.len() as f32;
        let mut cov = Array2::<f32>::zeros((2, 2));
        for w in &whitened {
            for i in 0..2 {
                for j in 0..2 {
                    cov[[i, j]] += w[i] * w[j];
                }
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let final_cov = cov / (n - 1.0);

        // Check diagonal (variance should be ~1.0)
        assert!((final_cov[[0, 0]] - 1.0).abs() < 1e-2);
        assert!((final_cov[[1, 1]] - 1.0).abs() < 1e-2);

        // Check off-diagonal (covariance should be ~0.0)
        assert!(final_cov[[0, 1]].abs() < 1e-2);
    }
}
