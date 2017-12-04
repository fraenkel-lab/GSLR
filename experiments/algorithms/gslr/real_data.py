#!/usr/bin/env python

#SBATCH --partition sched_mem1TB
#SBATCH --job-name=GSLR_GMM_PR
#SBATCH --output=/scratch/users/lenail/gslr/experiments/algorithms/gslr/multiprocess_%j.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=8000


import multiprocessing
import sys
import os

# necessary to add cwd to path when script run by slurm (since it executes a copy)
sys.path.append(os.getcwd())

# get number of cpus available to job
try:
	n_cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
	n_cpus = multiprocessing.cpu_count()

# ACTUAL APPLICATION LOGIC

import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import LabelEncoder

from matplotlib_venn import venn3, venn3_circles, venn2

repo_path = '/scratch/users/lenail/gslr/'
interactome_path = repo_path + 'experiments/algorithms/pcsf/inbiomap_temp.tsv'

sys.path.append(repo_path + 'gslr/')
import gslr



### V. Graph-Sparse Logistic Regression

def GSLR(X, y)

	d = len(nodes)
	c = 2

	graph_opts = gslr.GraphOptions(edges=edges, root=-1, num_clusters=1, pruning='strong')

	sparsity_low = 150
	sparsity_high = 400

	verbosity_level = 1

	num_steps = 100
	possible_steps = np.array([0.03, 0.1, 0.3])
	steps = np.tile(possible_steps, (num_steps, 1))

	W0 = np.zeros((c, d))

	W_hat, losses = gslr.gslr(X, y, W0, sparsity_low, sparsity_high, graph_opts, steps, verbosity_level, edge_costs=inbiomap_experimentally.cost.values, edge_costs_multiplier=6)

	yhat = gslr.predict(X, W_hat)
	num_cor = gslr.num_correct(y, yhat)

	return num_cor, W_hat, losses


if __name__ == "__main__":


	### I. Load Ovarian Cancer Proteomics Dataset

	# medullo = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/medullo_inbiomap_exp.tsv', index_col=0)
	dataset = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/ovarian_inbiomap_exp.tsv', index_col=0)
	# brca = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/brca_inbiomap_exp.tsv', index_col=0)

	# medullo_labels = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/raw/medullo_labels.csv', index_col=0)
	labels = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/raw/ovarian_labels.csv', index_col=0)
	# brca_labels = pd.read_csv('/Users/alex/Documents/proteomics/data_preparation/proteomics_data/raw/brca_labels.csv', index_col=0)


	### II. Load Interactome

	inbiomap_experimentally = pd.read_csv(interactome_path, sep='\t', names=['protein1','protein2','cost'])
	inbiomap_experimentally.head()


	(edges, nodes) = pd.factorize(inbiomap_experimentally[["protein1","protein2"]].unstack())
	edges = edges.reshape(inbiomap_experimentally[["protein1","protein2"]].shape, order='F')


	### IV. Prepare Dataset

	dataset = dataset.transpose().reindex(index=nodes).transpose()
	X = dataset.values


	labels = labels.values.flatten().tolist()


	labeler = LabelEncoder()
	labeler.fit(labels)
	y = labeler.transform(labels)





