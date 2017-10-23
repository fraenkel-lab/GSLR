# Synthetic Data Generation Strategy #1: "Multivariate Normal"

This was our first try at generating semi-synthetic data.
We went through a number of iterations, and did not use this code in the final paper.

## Outline:

1. Get means and covariance from empirical data
2. Get pathway information from KEGG.
3. Sample from multivariate normal defined by empirical data
4. For "positive" examples, shift mean values of proteins in pathway by a sample from their univariate normals in the empirical data


## Reason this did not work:

With a "short, fat" (n << p) matrix with more features than examples, the covariance matrix is
not positive semi-definite, which causes the data generation procedure to alternately take hours
or not work at all.

