#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=DATA_GEN
#SBATCH --output=/home/lenail/proteomics/synthetic_proteomics/data_generation/ludwig_svd_covariance_gaussian_mixture/multiprocess_%j.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=8000


import multiprocessing
import sys
import os

from functools import partial

# necessary to add cwd to path when script run by slurm (since it executes a copy)
sys.path.append(os.getcwd())

# get number of cpus available to job
try:
	n_cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
	n_cpus = multiprocessing.cpu_count()

# ACTUAL APPLICATION LOGIC

import numpy as np
import pandas as pd
import pickle as pkl
import math


# data is n x d, where n is the number of samples and d the dimension
def sample_cov(m, data):
	n, d = data.shape
	mean = np.mean(data, axis=0)
	_, S, V = np.linalg.svd((data - mean) / math.sqrt(n), full_matrices=False)
	randomness = np.random.randn(m, min(n, d))
	return np.dot(randomness, np.dot(np.diag(S), V))


def generate_dataset(pathway_id, pathway_genes, proteomics, POSITIVE_SAMPLES=100, NEGATIVE_SAMPLES=100):

	print(pathway_id)

	means = proteomics.mean(axis=0)
	variances = proteomics.var(axis=0)
	stddev = proteomics.std(axis=0)

	negatives = sample_cov(100, proteomics)
	negatives = np.around(negatives + means.values, 6)
	negatives = pd.DataFrame(negatives, columns=proteomics.columns, index=['negative']*100)

	shift_direction = np.random.randint(2, size=means.shape)*2-1  # vecrtor of +1/-1, "hack"
	new_pathway_means = pd.Series(np.random.normal(stddev*shift_direction,variances), index=variances.index)[pathway_genes].fillna(0)
	new_means = pd.concat([means, new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)

	positives = sample_cov(100, proteomics)
	positives = np.around(positives + new_means.values, 6)
	positives = pd.DataFrame(positives, columns=proteomics.columns, index=[pathway_id]*100)

	dataset = pd.concat([positives, negatives]).sample(frac=1)  # shuffle

	filename = '/home/lenail/proteomics/synthetic_proteomics/generated_data/ludwig_svd_normals_gaussian_mixture/'+pathway_id+'_inbiomap_exp.csv'
	return dataset.to_csv(filename, index=True, header=True)



if __name__ == "__main__":

	ovarian = pd.read_csv('/home/lenail/proteomics/data_preparation/proteomics_data/ovarian_inbiomap_exp.tsv', index_col=0)

	pathways_df = pd.read_pickle('/home/lenail/proteomics/synthetic_proteomics/data_generation/KEGG_df.filtered.with_correlates.pickle')

	pathways = [(pathway_id, pathways_df.loc[pd.IndexSlice[:, :, [pathway_id]],['genes', 'correlates']].values[0][0]) for pathway_id in pathways_df.index.get_level_values(2)]

	def generate_dataset_func(pathway): generate_dataset(pathway[0], np.unique(pathway[1]), ovarian)

	pool = multiprocessing.Pool(n_cpus)
	pool.map(generate_dataset_func, pathways)


