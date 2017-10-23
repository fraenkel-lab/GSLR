# Synthetic Data Generation Strategy #5: "Sampling from a Conditional Multivariate Normal"

The goal here was to do the best possible job of matching our intuition in the biology.
Sadly, we didn't have time to implement this fully.

## Outline:

1. Get means and covariance from empirical data
2. Get pathway information from KEGG.
3. Sample **negative exmples only** from multivariate normal defined by empirical data via SVD trick
4. For "positive" examples, shift mean values of proteins in pathway by a sample from their univariate normals in the empirical data
5. Sample the positive examples from a conditional multivariate gaussian where the proteins not in the pathway are sampled taking into account the values of the proteins in the pathway.

## Reason this did not work:

From the biological perspective, we had sets of proteins (pathways) we described as ground truth.
But if another protein covaries strongly with a pathway, in some sense, it is also part of that pathway.
It would likely needlessly hurt our algorithmic performance to include effects on these other proteins,
as well as the performances of all the other algorithms.

We also didn't want to get bogged down in writing more and more data generation strategies and wanted to move on to writing our algorithm!
