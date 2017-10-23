#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=LOGIT
#SBATCH --output=/home/lenail/gslr/experiments/algorithms/multiprocess_%j.out
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

from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, cross_val_score, KFold


def logit(filepath_and_pathway_ids):

	filepath, first_pathway_id, second_pathway_id = filepath_and_pathway_ids

	# we had done dataset.to_csv(filename, index=True, header=True)
	dataset = pd.read_csv(filepath, index_col=0)
	labels = dataset.index.str.replace(first_pathway_id, "positive").str.replace(second_pathway_id, "positive").tolist()

	classifier = LogisticRegressionCV(solver='liblinear', penalty='l1', Cs=[5], cv=10)
	classifier.fit(dataset.values, labels)
	features = pd.DataFrame(classifier.coef_, columns=dataset.columns)
	features = features.ix[0, features.loc[0].nonzero()[0].tolist()].index.tolist()
	scores = list(classifier.scores_.values())[0].flatten().tolist()

	return first_pathway_id, second_pathway_id, scores, features

# oncogenes = {subtype: ovarian_coefs.ix[subtype, ovarian_coefs.loc[subtype].nonzero()[0].tolist()].index.tolist() for subtype in ovarian_coefs.index.tolist()}
# oncogenes_pairs = {subtype: list(combinations(genes, r=2)) for subtype, genes in oncogenes.items()}
# path_lengths = {subtype: [nx.shortest_path_length(interactome, source=pair[0], target=pair[1]) for pair in pairs if pair[0] in interactome.nodes() and pair[1] in interactome.nodes()] for subtype, pairs in oncogenes_pairs.items()}
# [pd.Series(path_lengths[subtype]).plot.hist() for subtype in path_lengths]


if __name__ == "__main__":

	repo_path = '/home/lenail/gslr/experiments/'
	data_path = repo_path + 'generated_data/xor_3/'
	KEGG_path = repo_path + 'KEGG/KEGG_df.filtered.with_correlates.pickle'
	interactome_path = repo_path + 'algorithms/pcsf/inbiomap_temp.tsv'
	pathways_df = pd.read_pickle(KEGG_path)

	files = [(pathway_id, data_path+pathway_id+'_inbiomap_exp.csv') for pathway_id in pathways_df.index.get_level_values(2)]

	pool = multiprocessing.Pool(n_cpus)
	results = pool.map(logit, files)

	pickle.dump(results, open('xor_logit_results.pickle', 'wb'))

