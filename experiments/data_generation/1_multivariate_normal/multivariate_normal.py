#!/usr/bin/env python

#SBATCH --partition sched_mit_hill
#SBATCH --constraint=centos7
#SBATCH --job-name=KEGG
#SBATCH --output=/home/lenail/proteomics/synthetic_proteomics/multiprocess_%j.out
#SBATCH -n 8
#SBATCH -N 1
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

# ovarian = pd.read_csv('../data_preparation/ovarian_inbiomap_exp.tsv', index_col=0)

# means = ovarian.mean(axis=0)
# covariances = ovarian.cov()
# variances = ovarian.var()

pathways = pkl.load(open("KEGG_pathway_gene_lists.pkl", "rb"))


def generate_dataset(pathway):

	pathway_id, pathway_genes = pathway

	POSITIVE_SAMPLES = 100
	NEGATIVE_SAMPLES = 100

	ovarian = pd.read_csv('../data_preparation/ovarian_inbiomap_exp.tsv', index_col=0)

	means = ovarian.mean(axis=0)
	covariances = ovarian.cov()
	variances = ovarian.var()

	print('here')

	new_pathway_means = pd.Series(np.random.normal(0,variances), index=variances.index)[pathway_genes].fillna(0)
	new_means = pd.concat([means, new_pathway_means], axis=1).fillna(0).sum(axis=1).reindex(means.index)

	positives = pd.DataFrame(np.random.multivariate_normal(new_means, covariances, size=POSITIVE_SAMPLES))
	positives.index = [pathway_id+' positive']*len(positives)

	negatives = pd.DataFrame(np.random.multivariate_normal(means, covariances, size=NEGATIVE_SAMPLES))
	negatives.index = [pathway_id+' negative']*len(negatives)

	dataset = pd.concat([positives, negatives]).sample(frac=1)  # shuffle
	dataset.columns = ovarian.columns

	filename = 'synthetic_'+pathway_id+'_'+str(POSITIVE_SAMPLES)+'pos_'+str(NEGATIVE_SAMPLES)+'neg.csv'
	return dataset.to_csv(filename, index=True, header=True)


if __name__ == "__main__":

	# create pool of n_cpus workers
	pool = multiprocessing.Pool(n_cpus)
	pool.map(generate_dataset, list(pathways.items())[:10])



## APPLY FUNCTION IN PARALLEL

# pool.starmap(generate_dataset_func, pathways.items())



# Greg's phone: 617 797 4664


