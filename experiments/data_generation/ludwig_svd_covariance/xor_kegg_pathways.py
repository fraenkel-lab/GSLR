#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=KEGG
#SBATCH --output=/home/lenail/sampling/multiprocess_%j.out
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

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

# data is n x d, where n is the number of samples and d the dimension
def sample_cov(m, data):
	n, d = data.shape
	mean = np.mean(data, axis=0)
	_, S, V = np.linalg.svd((data - mean) / math.sqrt(n), full_matrices=False)
	randomness = np.random.randn(m, min(n, d))
	return np.dot(randomness, np.dot(np.diag(S), V))


def generate_dataset(first_pathway_id, first_pathway_genes, second_pathway_id, second_pathway_genes, proteomics, POSITIVE_SAMPLES=100, NEGATIVE_SAMPLES=100):

	means = proteomics.mean(axis=0)
	variances = proteomics.var(axis=0)

	negatives = sample_cov(50, proteomics)
	negatives = np.around(negatives + means.values, 6)
	negatives = pd.DataFrame(negatives, columns=proteomics.columns, index=['negative']*50)

	first_new_pathway_means = pd.Series(np.random.normal(0,variances), index=variances.index)[first_pathway_genes].fillna(0)
	second_new_pathway_means = pd.Series(np.random.normal(0,variances), index=variances.index)[second_pathway_genes].fillna(0)

	first_new_means = pd.concat([means, first_new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)
	second_new_means = pd.concat([means, second_new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)
	both_new_means = pd.concat([means, first_new_pathway_means, second_new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)

	first = sample_cov(50, proteomics)
	first = np.around(first + first_new_means.values, 6)
	first = pd.DataFrame(first, columns=proteomics.columns, index=[first_pathway_id]*50)

	second = sample_cov(50, proteomics)
	second = np.around(second + second_new_means.values, 6)
	second = pd.DataFrame(second, columns=proteomics.columns, index=[second_pathway_id]*50)

	both = sample_cov(50, proteomics)
	both = np.around(both + both_new_means.values, 6)
	both = pd.DataFrame(both, columns=proteomics.columns, index=['negative']*50)

	dataset = pd.concat([negatives,first,second,both]).sample(frac=1)  # shuffle

	filename = './xor_ludwig_svd_normals/'+first_pathway_id+'_'+second_pathway_id+'_inbiomap_exp.csv'
	return dataset.to_csv(filename, index=True, header=True)


if __name__ == "__main__":

	ovarian = pd.read_csv('ovarian_inbiomap_exp.tsv', index_col=0)

	pathways = pkl.load(open("KEGG_pathway_gene_lists.pkl", "rb"))

	pathway_pairs = pairwise(pathways.items())

	def generate_dataset_func(pathway_pair): generate_dataset(pathway_pair[0][0], np.unique(pathway_pair[0][1]), pathway_pair[1][0], np.unique(pathway_pair[1][1]), ovarian)

	pool = multiprocessing.Pool(n_cpus)
	pool.map(generate_dataset_func, pathway_pairs)


