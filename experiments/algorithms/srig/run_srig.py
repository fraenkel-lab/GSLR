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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split, cross_val_score, KFold

import spams
import scipy.sparse as ssp


def srig_predict(X, W): return np.argmax(np.dot(X, W.transpose()), axis=1)

def num_correct(y1, y2): return np.count_nonzero(np.equal(y1, y2))

def SRIG(pathway_id_and_filepath_and_nodes_and_edges_and_costs):

    pathway_id, filepath, nodes, edges, costs = pathway_id_and_filepath_and_nodes_and_edges_and_costs


    # we had done dataset.to_csv(filename, index=True, header=True)
    dataset = pd.read_csv(filepath, index_col=0)
    y = LabelEncoder().fit_transform(dataset.index.tolist())

    dataset = dataset.transpose().reindex(index=nodes).transpose()
    X = dataset.values


    neighborhoods = [[nodes.get_loc(node)]+[nodes.get_loc(neighbor) for neighbor in inbiomap_experimentally_graph.neighbors(node)] for node in nodes]
    num_groups = len(neighborhoods)

    # Name: spams.fistaGraph
    #
    # Description:
    #     spams.fistaGraph solves sparse regularized problems.
    #         X is a design matrix of size m x p
    #         X=[x^1,...,x^n]', where the x_i's are the rows of X
    #         Y=[y^1,...,y^n] is a matrix of size m x n
    #
    #         It implements the algorithms FISTA, ISTA and subgradient descent for solving
    #
    #           min_W  loss(W) + lambda1 psi(W)
    #
    #         The function psi are those used by spams.proximalGraph (see documentation)
    #         for the loss functions, see the documentation of spams.fistaFlat
    #
    #         This function can also handle intercepts (last row of W is not regularized),
    #         and/or non-negativity constraints on W.
    #

    #       graph: struct
    #             with three fields, eta_g, groups, and groups_var
    #
    #             The first fields sets the weights for every group
    #                graph.eta_g            double N vector

    eta_g = np.ones(num_groups)

    #             The next field sets inclusion relations between groups (but not between groups and variables):
    #                graph.groups           sparse (double or boolean) N x N matrix
    #                the (i,j) entry is non-zero if and only if i is different than j and
    #                gi is included in gj.

    groups = scipy.sparse.csc_matrix(np.zeros((num_groups,num_groups)),dtype=np.bool)


    i, j = zip(*flatten([[(i, j) for j in neighbors] for i, neighbors in enumerate(neighborhoods)]))


    #             The next field sets inclusion relations between groups and variables
    #                graph.groups_var       sparse (double or boolean) p x N matrix
    #                the (i,j) entry is non-zero if and only if the variable i is included
    #                in gj, but not in any children of gj.

    #  scipy.sparse.csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    #      where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].

    groups_var = scipy.sparse.csc_matrix((np.ones(len(i)),(i,j)),dtype=np.bool)

    #       graph: struct
    #             with three fields, eta_g, groups, and groups_var
    #
    graph = {'eta_g':eta_g,'groups':groups,'groups_var':groups_var}

    # Usage: spams.fistaGraph(  Y,
    #                           X,
    #                           W0,
    #                           graph,
    #                           return_optim_info=False,
    #                           numThreads=-1,
    #                           max_it=1000,
    #                           L0=1.0,
    #                           fixed_step=False,
    #                           gamma=1.5,
    #                           lambda1=1.0,
    #                           lambda2=0.,
    #                           lambda3=0.,
    #                           a=1.0,
    #                           b=0.,
    #                           tol=0.000001,
    #                           it0=100,
    #                           compute_gram=False,
    #                           intercept=False,
    #                           regul="",
    #                           loss="",
    #                           verbose=False,
    #                           pos=False,
    #                           ista=False,
    #                           subgrad=False,
    #                           linesearch_mode=0)
    #
    # Inputs:
    #       Y                     : double dense m x n matrix

    Y = np.asfortranarray(np.expand_dims(y, axis=1)).astype(float)
    Y = spams.normalize(Y)

    #       X                     : double dense or sparse m x p matrix

    X = np.asfortranarray(dataset.values).astype(float)
    X = spams.normalize(X)

    #       W0                    : double dense p x n matrix or p x Nn matrix for multi-logistic loss initial guess

    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="F")

    #       graph                 : struct see documentation of proximalGraph
    #       return_optim_info     : if true the function will return a tuple of matrices.
    #       loss                  : choice of loss, see above
    #       regul                 : choice of regularization, see below
    #       lambda1               : regularization parameter
    #       lambda2               : regularization parameter, 0 by default
    #       lambda3               : regularization parameter, 0 by default
    #       verbose               : verbosity level, false by default
    #       pos                   : adds positivity constraints on the coefficients, false by default
    #       numThreads            : number of threads for exploiting multi-core / multi-cpus. By default, it takes the value -1, which automatically selects all the available CPUs/cores.
    #       max_it                : maximum number of iterations, 100 by default
    #       it0                   : frequency for computing duality gap, every 10 iterations by default
    #       tol                   : tolerance for stopping criteration, which is a relative duality gap if it is available, or a relative change of parameters.
    #       gamma                 : multiplier for increasing the parameter L in fista, 1.5 by default
    #       L0                    : initial parameter L in fista, 0.1 by default, should be small enough
    #       fixed_step            : deactive the line search for L in fista and use L0 instead
    #       compute_gram          : pre-compute X^TX, false by default.
    #       intercept             : do not regularize last row of W, false by default.
    #       ista                  : use ista instead of fista, false by default.
    #       subgrad               : if not ista, use subradient descent instead of fista, false by default.
    #       a                     :
    #       b                     : if subgrad, the gradient step is a/(t+b) also similar options as proximalTree

    loss         = 'square'
    regul        = 'graph'
    lambda1      = 0.1
    L0           = 0.1
    verbose      = True

    features = []
    accuracies = []

    for train, test in StratifiedKFold(n_splits=10).split(X, y):

        W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="F")

        (W, optim_info) = spams.fistaGraph(Y, X, W0, graph, loss=loss, regul=regul, lambda1=lambda1, return_optim_info=True, verbose=verbose)

        yhat = srig_predict(X[test], W_hat)
        num_cor = num_correct(y[test], yhat)
        accuracy = num_cor / float(len(test))

        features.append(W_hat[0])
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
    (edges, nodes) = pd.factorize(inbiomap_experimentally[["protein1","protein2"]].unstack())
    edges = edges.reshape(inbiomap_experimentally[["protein1","protein2"]].shape, order='F')
    costs = inbiomap_experimentally.cost.values

    inputs = [(pathway_id, data_path+pathway_id+'_inbiomap_exp.csv', nodes, edges, costs) for pathway_id in pathways_df.index.get_level_values(2)]

    pool = multiprocessing.Pool(n_cpus)

    results = pool.map(SRIG, inputs)

    pickle.dump(results, open('srig_results.pickle', 'wb'))






