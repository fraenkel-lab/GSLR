#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=srig
#SBATCH --output=/home/lenail/srig/experiments/algorithms/srig/multiprocess_%j.out
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
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import spams
import scipy.sparse as ssp


def srig_predict(X, W): return np.argmax(np.dot(X, W.transpose()), axis=1)

def num_correct(y1, y2): return np.count_nonzero(np.equal(y1, y2))

def SRIG(pathway_id_and_filepath_and_graph_struct_and_lambda):

    pathway_id, filepath, graph, lambda1 = pathway_id_and_filepath_and_graph_struct_and_lambda

    print()
    print('-----------------')
    print(pathway_id)
    print(str(sparsity_low)+'-'+str(sparsity_high))
    print()

    # we had done dataset.to_csv(filename, index=True, header=True)
    dataset = pd.read_csv(filepath, index_col=0)
    y = LabelEncoder().fit_transform(dataset.index.tolist())
    Y = np.asfortranarray(np.expand_dims(y, axis=1)).astype(float)
    Y = spams.normalize(Y)

    dataset = dataset.transpose().reindex(index=nodes).transpose()
    X = dataset.values
    X = np.asfortranarray(dataset.values).astype(float)
    X = spams.normalize(X)

    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="F")

    features = []
    accuracies = []

    for train, test in StratifiedKFold(n_splits=10).split(X, y):

        print()
        print('fold')
        print()

        W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="F")

        (W, optim_info) = spams.fistaGraph(Y, X, W0, graph, loss='square', regul='graph', lambda1=lambda1, return_optim_info=True)

        yhat = srig_predict(X[test], W)
        num_cor = num_correct(y[test], yhat)
        accuracy = num_cor / float(len(test))

        features.append(W)
        accuracies.append(accuracy)

    features = pd.DataFrame(features, columns=dataset.columns)
    features = features.columns[(features != 0).any()].tolist()

    return pathway_id, accuracies, features


if __name__ == "__main__":

    repo_path = '/home/lenail/gslr/experiments/'
    data_path = repo_path + 'generated_data/3/'
    KEGG_path = repo_path + 'KEGG/KEGG_df.filtered.with_correlates.pickle'
    interactome_path = repo_path + 'algorithms/pcsf/inbiomap_temp.tsv'
    pathways_df = pd.read_pickle(KEGG_path)

    inbiomap_experimentally = pd.read_csv(interactome_path, sep='\t', names=['protein1','protein2','cost'])
    inbiomap_experimentally_graph = nx.from_pandas_edgelist(inbiomap_experimentally, 'protein1', 'protein2', edge_attr=True)
    (edges, nodes) = pd.factorize(inbiomap_experimentally[["protein1","protein2"]].unstack())
    edges = edges.reshape(inbiomap_experimentally[["protein1","protein2"]].shape, order='F')

    neighborhoods = [[nodes.get_loc(node)]+[nodes.get_loc(neighbor) for neighbor in inbiomap_experimentally_graph.neighbors(node)] for node in nodes]
    num_groups = len(neighborhoods)

    eta_g = np.ones(num_groups)

    groups = scipy.sparse.csc_matrix(np.zeros((num_groups,num_groups)),dtype=np.bool)

    i, j = zip(*flatten([[(i, j) for j in neighbors] for i, neighbors in enumerate(neighborhoods)]))
    groups_var = scipy.sparse.csc_matrix((np.ones(len(i)),(i,j)),dtype=np.bool)

    graph = {'eta_g':eta_g,'groups':groups,'groups_var':groups_var}


    inputs = [(pathway_id, data_path+pathway_id+'_inbiomap_exp.csv', graph) for pathway_id in pathways_df.index.get_level_values(2)]

    pool = multiprocessing.Pool(n_cpus)

    results = pool.map(SRIG, inputs)

    pickle.dump(results, open('/scratch/users/lenail/results/srig_pr_results.pickle', 'wb'))






