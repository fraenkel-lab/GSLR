# Synthetic Data Generation Strategy #4: "Multivariate Normals with the SVD trick and greater perturbation"

In this case, we do the exact same thing as Strategy #3 but the means of the proteins in the pathways are shifted more.

## Outline:

1. Get means and covariance from empirical data
2. Get pathway information from KEGG.
3. Sample from multivariate normal defined by empirical data via SVD trick
4. For "positive" examples, shift mean values of proteins in pathway by **one standard deviation** of their univariate normals in the empirical data

### This was one of the strategies we used in the final paper


