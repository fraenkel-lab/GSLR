# KEGG database download and filtering

`download_KEGG.ipynb` downloads the KEGG database as a dataframe.

`KEGG_df.pickle` is the resulting dataframe, the version we used for our experiments. KEGG has likely been updated since, so get the newest version. However, if you'd like to reproduce our results exactly, you will need to use this older version from mid-2017.

`Pathway_Curation_and_Correlates_Discovery.ipynb` filters out the metabolic pathways from `KEGG_df.pickle`, which are poorly connected in the interactomes because proteins which belong to the same metabolic pathways infrequently need to bind to one another. This notebook also appends to each pathway the top 100 correlates from our empirical data, which we never ended up using.

`KEGG_df.filtered.with_correlates.pickle` is the resulting dataframe. This is the exact dataframe we used for all our experiments.

