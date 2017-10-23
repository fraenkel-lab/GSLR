import numpy as np
import pandas as pd
import pickle as pkl


def generate_dataset(pathway_id, pathway_genes, ovarian, POSITIVE_SAMPLES=100, NEGATIVE_SAMPLES=100):

	means = ovarian.mean(axis=0)
	variances = ovarian.var()

	new_pathway_means = pd.Series(np.random.normal(0,variances), index=variances.index)[pathway_genes].fillna(0)
	new_means = pd.concat([means, new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)

	positives = pd.DataFrame(np.random.normal(new_means, variances, size=(100, len(means))), columns=ovarian.columns)
	positives.index = [pathway_id]*len(positives)

	negatives = pd.DataFrame(np.random.normal(means, variances, size=(100, len(means))), columns=ovarian.columns)
	negatives.index = ['negative']*len(negatives)

	dataset = pd.concat([positives, negatives]).sample(frac=1)  # shuffle

	filename = './independent_normals/'+pathway_id+'.csv'
	return dataset.to_csv(filename, index=True, header=True)


if __name__ == "__main__":

	ovarian = pd.read_csv('../data_preparation/ovarian_inbiomap_exp.tsv', index_col=0)

	pathways = pkl.load(open("KEGG_pathway_gene_lists.pkl", "rb"))

	for pathway_id, pathway_genes in list(pathways.items())[:1]:

		generate_dataset(pathway_id, pathway_genes, ovarian)


