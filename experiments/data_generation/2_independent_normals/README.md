# Synthetic Data Generation Strategy #2: "Independent Normals"

We briefly entertained the idea of sampling semi-synthetic data from the univariate
normals for each feature in the empirical data.

## Outline:

1. Get means and variances from empirical data
2. Get pathway information from KEGG.
3. Change mean values of proteins in pathway
4. Sample from **independent normals** for each protein


## Reason this did not work:

The gaussian of patient proteomics is not a perfect hypershphere.

