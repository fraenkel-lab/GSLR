#!/usr/bin/env python

#SBATCH --partition sched_mem1TB
#SBATCH --job-name=GSLR_PR
#SBATCH --output=/scratch/users/lenail/gslr/experiments/algorithms/gslr/multiprocess_%j.out
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=8000
#SBATCH --time=40:00:00


import multiprocessing
import sys
import os

import pickle
from itertools import product

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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split, cross_val_score, KFold

sys.path.append('/home/lenail/gslr/gslr')
import gslr

def GSLR(pathway_id_and_filepath_and_nodes_and_edges_and_costs_and_low_and_high):

	pathway_id, filepath, nodes, edges, costs, sparsity_low, sparsity_high = pathway_id_and_filepath_and_nodes_and_edges_and_costs_and_low_and_high

	print()
	print('-----------------')
	print(pathway_id)
	print(str(sparsity_low)+'-'+str(sparsity_high))
	print()

	# we had done dataset.to_csv(filename, index=True, header=True)
	dataset = pd.read_csv(filepath, index_col=0)
	y = LabelEncoder().fit_transform(dataset.index.tolist())

	dataset = dataset.transpose().reindex(index=nodes).transpose()
	X = dataset.values

	d = len(nodes)
	c = 2

	graph_opts = gslr.GraphOptions(edges=edges, root=-1, num_clusters=1, pruning='strong')

	verbosity_level = 1

	num_steps = 20
	possible_steps = np.array([0.03,0.1,0.3])
	steps = np.tile(possible_steps, (num_steps, 1))

	featuresets = []
	accuracies = []

	for train, test in StratifiedKFold(n_splits=10).split(X, y):

		print()
		print('fold')
		print()

		W0 = np.zeros((c, d))

		W_hat, losses = gslr.gslr(X[train], y[train], W0, sparsity_low, sparsity_high, graph_opts, steps, verbosity_level, edge_costs=costs, edge_costs_multiplier=2)

		yhat = gslr.predict(X[test], W_hat)
		num_cor = gslr.num_correct(y[test], yhat)
		accuracy = num_cor / float(len(test))
		accuracies.append(accuracy)

		features = pd.DataFrame(W_hat, columns=dataset.columns)
		features = features.columns[(features != 0).any()].tolist()
		featuresets.append(features)


	return pathway_id, (sparsity_low, sparsity_high), accuracies, featuresets


if __name__ == "__main__":

	repo_path = '/scratch/users/lenail/gslr/experiments/'
	data_path = repo_path + 'generated_data/3/'
	KEGG_path = repo_path + 'KEGG/KEGG_df.filtered.with_correlates.pickle'
	interactome_path = repo_path + 'algorithms/pcsf/inbiomap_temp.tsv'
	pathways_df = pd.read_pickle(KEGG_path)

	inbiomap_experimentally = pd.read_csv(interactome_path, sep='\t', names=['protein1','protein2','cost'])
	(edges, nodes) = pd.factorize(inbiomap_experimentally[["protein1","protein2"]].unstack())
	edges = edges.reshape(inbiomap_experimentally[["protein1","protein2"]].shape, order='F')
	costs = inbiomap_experimentally.cost.values

	lows_and_highs = [(0, 100),(50, 200),(150, 300),(250, 500),(500, 1500)]

	inputs = [(pathway_id, data_path+pathway_id+'_inbiomap_exp.csv', nodes, edges, costs, low, high) for pathway_id, (low, high) in product(pathways_df.index.get_level_values(2)[220:], lows_and_highs)]

	pool = multiprocessing.Pool(n_cpus)

	results = pool.map(GSLR, inputs)

	pickle.dump(results, open('/scratch/users/lenail/results/gslr_pr_results_220_229.pickle', 'wb'))








