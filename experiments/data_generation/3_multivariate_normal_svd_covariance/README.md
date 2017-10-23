# Synthetic Data Generation Strategy #3: "Multivariate Normals with the SVD trick"

Ludwig developed a routine to sample from a multivariate normal without needing
to invert the covariance through a clever SVD trick. This makes the original idea
tractable.

## Outline:

1. Get means and covariance from empirical data
2. Get pathway information from KEGG.
3. Sample from multivariate normal defined by empirical data **via SVD trick**
4. For "positive" examples, shift mean values of proteins in pathway by a sample from their univariate normals in the empirical data

### This was one of the strategies we used in the final paper


