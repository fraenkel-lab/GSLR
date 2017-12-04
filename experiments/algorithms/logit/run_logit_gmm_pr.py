#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=LOGIT_GMM_PR
#SBATCH --output=/scratch/users/lenail/gslr/experiments/algorithms/logit/multiprocess_%j.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=8000


import multiprocessing
import sys
import os

import pickle

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
from sklearn.linear_model import LogisticRegressionCV


def logit(pathway_id_and_filepath):

	pathway_id, filepath = pathway_id_and_filepath

	# we had done dataset.to_csv(filename, index=True, header=True)
	dataset = pd.read_csv(filepath, index_col=0)
	labels = dataset.index.tolist()

	classifier = LogisticRegressionCV(solver='liblinear', penalty='l2', Cs=16, cv=10)
	classifier.fit(dataset.values, labels)
	scores = list(classifier.scores_.values())

	# features = pd.DataFrame(W_hat, columns=dataset.columns)
	# features = features.columns[(features != 0).any()].tolist()

	return pathway_id, scores, classifier.coefs_paths_, classifier.Cs_


if __name__ == "__main__":

	repo_path = '/scratch/users/lenail/gslr/experiments/'
	data_path = repo_path + 'generated_data/4/'
	KEGG_path = repo_path + 'KEGG/KEGG_df.filtered.with_correlates.pickle'
	interactome_path = repo_path + 'algorithms/pcsf/inbiomap_temp.tsv'
	pathways_df = pd.read_pickle(KEGG_path)

	files = [(pathway_id, data_path+pathway_id+'_inbiomap_exp.csv') for pathway_id in pathways_df.index.get_level_values(2)]

	pool = multiprocessing.Pool(n_cpus)

	results = pool.map(logit, files)

	pickle.dump(results, open('/scratch/users/lenail/results/logit_l2_gmm_pr_results.pickle', 'wb'))

