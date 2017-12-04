#!/usr/bin/env python

# Core python modules
import sys
import os
import math

# Peripheral python modules
import collections

# python external libraries
import numpy as np
import numba

# lab modules
from pcst_fast import pcst_fast

__all__ = ["GraphOptions", "gslr", "softmax_gradient", "softmax_loss", "graph_proj_sparsity", "proj_softmax_matrix"]


@numba.jit
def softmax_gradient(X, y, W):
    """
    Computes the multi-class softmax gradient.

    The gradient is for the *average* loss.

    Arguments:
        X (np.array): the n x d data matrix (n examples in dimension d)
        y (np.array): the n-dimensional label vector. Each entry is an integer between 0 and c-1 (inclusive), where c is the number of classes.
        W (np.array): the c x d weight matrix

    Returns:
        (np.array): the gradient of W as a C x d matrix.
    """

    # TODO: make this work with nopython=True
    assert len(X.shape) == 2
    n, d = X.shape
    assert y.shape == (n,)

    assert len(W.shape) == 2
    assert W.shape[1] == d
    c = W.shape[0]
    # TODO: check that elements in y are in the correct range

    G = np.zeros_like(W)
    # TODO: replace with transpose_b=True ?
    prod = np.dot(X, W.transpose())
    row_max = np.amax(prod, axis=1)
    prod_normalized = prod - np.expand_dims(row_max, 1)
    exp_prod = np.exp(prod_normalized)
    denom = np.sum(exp_prod, axis=1)

    for ii in range(n):
        cur_y = y[ii]
        cur_x = X[ii,:]
        G[cur_y,:] += (1.0 - exp_prod[ii, cur_y] / denom[ii]) * cur_x
        for jj in range(c):
            if jj == cur_y:
                continue
            G[jj,:] -= exp_prod[ii, jj] / denom[ii] * cur_x

    return -G / n


@numba.jit
def softmax_loss(X, y, W):
    """
    Computes the softmax loss.

    Arguments:
        X (np.array): the n x d data matrix (n examples in dimension d)
        y (np.array): the n-dimensional label vector. Each entry is an integer between 0 and c-1 (inclusive), where c is the number of classes.
        W (np.array): the c x d weight matrix

    Returns:
        (np.array): the *average* loss as a scalar.
    """

    # TODO: make this work with nopython=True
    assert len(X.shape) == 2
    n, d = X.shape
    assert y.shape == (n,)

    assert len(W.shape) == 2
    assert W.shape[1] == d
    c = W.shape[0]
    # TODO: check that elements in y are in the correct range

    # TODO: replace with transpose_b=True ?
    prod = np.dot(X, W.transpose())
    row_max = np.amax(prod, axis=1)
    prod_normalized = prod - np.expand_dims(row_max, 1)
    exp_prod = np.exp(prod_normalized)
    denom = np.sum(exp_prod, axis=1)

    total_loss = 0.0
    for ii in range(n):
        cur_y = y[ii]
        cur_x = X[ii,:]
        total_loss += prod_normalized[ii, cur_y] - math.log(denom[ii])

    return -total_loss / n



GraphOptions = collections.namedtuple(
    'GraphOptions',
    ['edges', 'root', 'num_clusters', 'pruning'])


def graph_proj_sparsity(prizes, sparsity_low, sparsity_high, opts, verbosity_level, max_num_iter=30, edge_costs=None, edge_costs_multiplier=None):
    """
    Projects the input onto the graph sparsity model.

    Arguments:
        prizes (np.array): a real vector with non-negative node prizes (= parameter coefficients)
        sparsity_low (int): the (approximate) lower bound for the output sparsity
        sparsity_high (int): the (approximate) upper bound for the output sparsity
        opts (GraphOptions): passed directly to `pcst_fast`
        verbosity_level (int): indicates whether intermediate output should be printed verbosity_level - 1 is being passed to pcst_fast
        max_num_iter (int): maximum number of iterations
        edge_costs (np.array): a real vector with non-negative edge costs
        edge_costs_multiplier (np.array): a factor weighing edge costs vs prizes

    Returns:
        (np.array): the vector of graph-sparse prizes (0 for nodes not in the PCSF solution)
        (np.array): the list of indices of selected vertices
        (np.array): the list of indices of the selected edges
    """

    num_v, = prizes.shape
    num_e, _ = opts.edges.shape

    if edge_costs is not None:
        assert edge_costs_multiplier is not None
        costs = edge_costs_multiplier * edge_costs
    else:
        costs = np.ones(num_e)

    lambda_r = 0.0
    lambda_l = 3.0 * np.sum(prizes)
    min_nonzero_prize = 0.0
    for ii in range(num_v):
        if prizes[ii] > 0.0 and prizes[ii] < min_nonzero_prize:
            min_nonzero_prize = prizes[ii]
    eps = 0.01 * min_nonzero_prize

    if verbosity_level >= 2:
        print("Initial lambda_l = "+str(lambda_l)+"   lambda_r = "+str(lambda_r)+"   eps = "+str(eps))

    num_iter = 0
    while lambda_l - lambda_r > eps and num_iter < max_num_iter:
        num_iter += 1
        lambda_m = (lambda_l + lambda_r) / 2.0
        cur_vertices, cur_edges = pcst_fast(opts.edges, prizes, lambda_m * costs, opts.root, opts.num_clusters, opts.pruning, verbosity_level - 1)
        cur_sparsity = cur_vertices.size
        if verbosity_level >= 2:
            print("lambda_l = "+str(lambda_l)+"   lambda_r = "+str(lambda_r)+"   lambda_m = "+str(lambda_m)+"   cur_sparsity = "+str(cur_sparsity))
        if cur_sparsity >= sparsity_low and cur_sparsity <= sparsity_high:
            if verbosity_level >= 2:
                print('Returning intermediate solution for lambda_m')
            result = np.zeros_like(prizes)
            result[cur_vertices] = prizes[cur_vertices]
            return result, cur_vertices, cur_edges
        if cur_sparsity > sparsity_high:
            lambda_r = lambda_m
        else:
            lambda_l = lambda_m
    cur_vertices, cur_edges = pcst_fast(opts.edges, prizes, lambda_l * costs, opts.root, opts.num_clusters, opts.pruning, verbosity_level - 1)
    cur_sparsity = cur_vertices.size
    if cur_sparsity < sparsity_low:
        print("WARNING: returning sparsity "+str(cur_sparsity)+" although minimum sparsity "+str(sparsity_low)+" was requested.")
    if verbosity_level >= 2:
        print("Returning final solution for lambda_l (cur_sparsity = "+str(cur_sparsity)+")")
    result = np.zeros_like(prizes)
    result[cur_vertices] = prizes[cur_vertices]
    return result, cur_vertices, cur_edges


def proj_softmax_matrix(W, sparsity_low, sparsity_high, opts, verbosity_level, graph_proj_max_num_iter, edge_costs=None, edge_costs_multiplier=None):
    """


    Arguments:
        W (np.array): the c x d weight matrix
        sparsity_low (int): the (approximate) lower bound for the output sparsity
        sparsity_high (int): the (approximate) upper bound for the output sparsity
        opts (GraphOptions): passed directly to `pcst_fast`
        verbosity_level (int): indicates whether intermediate output should be printed verbosity_level - 1 is being passed to pcst_fast
        graph_proj_max_num_iter (int): the maximum number of iterations in the graph-sparsity projection.
        edge_costs (np.array): a real vector with non-negative edge costs
        edge_costs_multiplier (np.array): a factor weighing edge costs vs prizes

    Returns:
        (np.array): the graph-sparse c x d weight matrix
    """

    c, d = W.shape
    W2 = np.square(W)
    for ii in range(c):
        W2[ii,:], _, _ = graph_proj_sparsity(W2[ii,:], sparsity_low, sparsity_high, opts, verbosity_level, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
    return np.multiply(np.sign(W), np.sqrt(W2))


def gslr(X, y, W0, sparsity_low, sparsity_high, graph_opts, steps=None, verbosity_level=0, graph_proj_max_num_iter=20, edge_costs=None, edge_costs_multiplier=None):
    """
    Performs softmax regression with graph-sparsity constraints.

    Arguments:
        X (np.array): the n x d data matrix (n examples in dimension d)
        y (np.array): the n-dimensional label vector. Each entry is an integer between 0 and c-1 (inclusive), where c is the number of classes.
        W0 (np.array): the c x d weight matrix
        sparsity_low (int): the (approximate) lower bound for the output sparsity
        sparsity_high (int): the (approximate) upper bound for the output sparsity
        graph_opts (GraphOptions): passed directly to `pcst_fast`
        steps (np.array): the step size schedule, represented by a matrix of size num_steps x num_choices. In each iteration, the algorithm tries all current choices for the step size and chooses the one that makes largest progress.
        verbosity_level (int): indicates whether intermediate output should be printed verbosity_level - 1 is being passed to pcst_fast
        graph_proj_max_num_iter (int): the maximum number of iterations in the graph-sparsity projection.
        edge_costs (np.array): a real vector with non-negative edge costs
        edge_costs_multiplier (np.array): a factor weighing edge costs vs prizes

    Returns:
        (np.array): the final c x d weight matrix
        (np.array): the loss at each step, shape is (steps x 1)
    """

    assert len(steps.shape) == 2
    num_steps, num_choices = steps.shape

    losses = np.zeros(num_steps + 1)
    losses[0] = softmax_loss(X, y, W0)
    W_cur = np.copy(W0)
    for ii in range(num_steps):
        print("iteration "+str(ii+1)+":")
        gradients = softmax_gradient(X, y, W_cur)
        best_loss = losses[ii]
        best_step_size = 0.0
        # print("initially best loss "+str(best_loss))
        for step_size in steps[ii,:]:
            W_tmp = W_cur - step_size * gradients
            # print('before')
            # print(W_tmp)
            # print(softmax_loss(X, y, W_tmp))
            W_tmp = proj_softmax_matrix(W_tmp, sparsity_low, sparsity_high, graph_opts, verbosity_level, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
            # print('after')
            # print(W_tmp)
            # print(softmax_loss(X, y, W_tmp))
            loss_next = softmax_loss(X, y, W_tmp)
            if loss_next < best_loss:
                best_loss = loss_next
                best_step_size = step_size
            if verbosity_level >= 1:
                print("  loss_cur = "+str(losses[ii])+"   loss_next = "+str(loss_next)+"   step_size = "+str(step_size))
        if verbosity_level >= 1:
            print("  best_step_size: "+str(best_step_size))
        W_cur -= best_step_size * gradients
        W_cur = proj_softmax_matrix(W_cur, sparsity_low, sparsity_high, graph_opts, verbosity_level, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
        losses[ii + 1] = softmax_loss(X, y, W_cur)

    return W_cur, losses



# Helper functions
def predict(X, W):
    return np.argmax(np.dot(X, W.transpose()), axis=1)

def num_correct(y1, y2):
    return np.count_nonzero(np.equal(y1, y2))
