//! Hyperbolic Space Projections
//!
//! This module implements mappings from Euclidean space to Hyperbolic manifolds,
//! specifically the Poincaré Ball model, to better represent hierarchical data.

use crate::types::Embedding;
use serde::{Deserialize, Serialize};

/// Interface for vector space projections.
pub trait Projector {
    /// Projects a vector from the source space to the target manifold.
    fn project(&self, vector: &Embedding) -> Embedding;
}

/// A projector that maps Euclidean vectors into a Poincaré Ball (Hyperbolic space).
///
/// The mapping used is: `x_h` = tanh(||x||) * (x / ||x||)
/// This ensures all projected vectors reside within the unit ball (||`x_h`|| < 1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoincareProjector;

impl PoincareProjector {
    /// Creates a new instance of the `PoincareProjector`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Projector for PoincareProjector {
    fn project(&self, vector: &Embedding) -> Embedding {
        let norm = vector.dot(vector).sqrt();

        if norm < 1e-10 {
            // Avoid division by zero for origin vectors
            return vector.clone();
        }

        let scale = norm.tanh() / norm;
        vector * scale
    }
}

/// A projector that reduces dimensionality using the Johnson-Lindenstrauss lemma.
/// It utilizes a random Gaussian matrix to preserve pairwise distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomProjector {
    /// The projection matrix of shape (`target_dim`, `source_dim`).
    pub matrix: ndarray::Array2<f32>,
}

impl RandomProjector {
    /// Creates a new `RandomProjector` with a Gaussian random matrix.
    ///
    /// # Arguments
    /// * `source_dim` - Original dimensionality.
    /// * `target_dim` - Target dimensionality.
    ///
    /// # Panics
    /// Panics if the internal Gaussian distribution parameters are invalid.
    #[must_use]
    pub fn new_gaussian(source_dim: usize, target_dim: usize) -> Self {
        use rand::distributions::Distribution;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        // JL Projection scale: 1/sqrt(target_dim)
        #[allow(clippy::cast_precision_loss)]
        let std_dev = 1.0 / (target_dim as f32).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();

        let matrix =
            ndarray::Array2::from_shape_fn((target_dim, source_dim), |_| dist.sample(&mut rng));

        Self { matrix }
    }
}

impl Projector for RandomProjector {
    fn project(&self, vector: &Embedding) -> Embedding {
        self.matrix.dot(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_poincare_projection_unit_bound() {
        let projector = PoincareProjector::new();

        // Test with a very large vector
        let v = Array1::from_vec(vec![100.0, 100.0, 100.0]);
        let projected = projector.project(&v);

        let norm = projected.dot(&projected).sqrt();
        // Norm must be <= 1.0 (tanh(x) < 1)
        assert!(norm < 1.0);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_poincare_zero_vector() {
        let projector = PoincareProjector::new();
        let v = Array1::zeros(3);
        let projected = projector.project(&v);
        assert!(projected.mapv(f32::abs).sum() < 1e-10);
    }

    #[test]
    fn test_poincare_direction_preservation() {
        let projector = PoincareProjector::new();
        let v = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let projected = projector.project(&v);

        // Direction should be the same, magnitude should be tanh(1)
        assert!((projected[0] - 1.0f32.tanh()).abs() < 1e-6);
        assert!(projected[1].abs() < 1e-6);
        assert!(projected[2].abs() < 1e-6);
    }

    #[test]
    fn test_random_projection_dimensions() {
        let source_dim = 10;
        let target_dim = 3;
        let projector = RandomProjector::new_gaussian(source_dim, target_dim);

        let v = Array1::zeros(source_dim);
        let projected = projector.project(&v);

        assert_eq!(projected.len(), target_dim);
    }

    #[test]
    fn test_random_projection_jl_distance_preservation() {
        // Simple test to check if distance is roughly preserved
        let source_dim = 100;
        let target_dim = 50;
        let projector = RandomProjector::new_gaussian(source_dim, target_dim);

        let v1 = Array1::from_elem(source_dim, 1.0);
        let v2 = Array1::from_elem(source_dim, 0.0);

        let diff_orig = &v1 - &v2;
        let dist_orig_sq = diff_orig.dot(&diff_orig);

        let p1 = projector.project(&v1);
        let p2 = projector.project(&v2);

        let diff_proj = &p1 - &p2;
        let dist_proj_sq = diff_proj.dot(&diff_proj);

        // JL lemma guarantees (1-eps)d < d_proj < (1+eps)d
        // With 100->50, we expect some variance but it should be "close"
        let ratio = dist_proj_sq / dist_orig_sq;
        assert!(ratio > 0.5 && ratio < 1.5);
    }
}
