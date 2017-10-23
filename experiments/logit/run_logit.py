#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=LOGIT
#SBATCH --output=/home/lenail/proteomics/synthetic_proteomics/analysis/logit/multiprocess_%j.out
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


def logit(pathway_id_and_filepath):

	pathway_id, filepath = pathway_id_and_filepath

	# we had done dataset.to_csv(filename, index=True, header=True)
	dataset = pd.read_csv(filepath, index_col=0)
	labels = dataset.index.tolist()

	classifier = LogisticRegressionCV(solver='liblinear', penalty='l1', Cs=[5], cv=10)
	classifier.fit(dataset.values, labels)
	features = pd.DataFrame(classifier.coef_, columns=dataset.columns)
	features = features.ix[0, features.loc[0].nonzero()[0].tolist()].index.tolist()
	scores = list(classifier.scores_.values())

	return pathway_id, scores, features


if __name__ == "__main__":

	project_root = '/home/lenail/proteomics/synthetic_proteomics/'

	pathways_df = pd.read_pickle(project_root+'data_generation/KEGG_df.filtered.with_correlates.pickle')

	files = [(pathway_id, project_root+'generated_data/ludwig_svd_normals/'+pathway_id+'_inbiomap_exp.csv') for pathway_id in pathways_df.index.get_level_values(2)]


	pool = multiprocessing.Pool(n_cpus)

	results = pool.map(logit, files)

	pickle.dump(results, open('logit_results.pickle', 'wb'))

