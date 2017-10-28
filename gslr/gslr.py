import collections
import math
import numpy as np
# import numba

import pcst_fast

# Computes the softmax gradient for the following inputs:
# X is the n x d data matrix (n examples in dimension d)
# y is the n-dimensional label vector.
#   Each entry is an integer between 0 and c-1 (inclusive), where c is the number of classes.
# W is the c x d weight matrix
# The function returns the gradient of W as a C x d matrix.
# The gradient is for the *average* loss.
@numba.jit
def softmax_gradient(X, y, W):
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

# Computes the softmax loss for the following inputs:
# X is the n x d data matrix (n examples in dimension d)
# y is the n-dimensional label vector.
#   Each entry is an integer between 0 and c-1 (inclusive), where c is the number of classes.
# W is the c x d weight matrix
# The function returns the *average* loss as a scalar.
@numba.jit
def softmax_loss(X, y, W):
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
# Projects the input onto the graph sparsity model.
# prizes is a real vector with non-negative node prizes
#     (= parameter coefficients)
# sparsity_low and sparsity_high are the (approximate) upper and lower bounds
#     for the output sparsity
# opts.edges, opts.root, num_clusters, and opts.pruning are directly passed
#     to pcst_fast
# verbosity_level indicates whether intermediate output should be printed
#     verbosity_level - 1 is being passed to pcst_fast
# edge_costs is a real vector with non-negative edge costs
# edge_costs_multiplier: a factor weighing edge costs vs prizes
def graph_proj_sparsity(prizes, sparsity_low, sparsity_high, opts, verbosity_level, max_num_iter=30, edge_costs=None, edge_costs_multiplier=None):
    num_v, = prizes.shape
    num_e, _ = opts.edges.shape
    costs1 = np.ones(num_e)
    if edge_costs is not None:
        assert edge_costs_multiplier is not None
        costs = costs1 + edge_costs_multiplier * edge_costs
    else:
        costs = costs1
    lambda_r = 0.0
    lambda_l = 3.0 * np.sum(prizes)
    min_nonzero_prize = 0.0
    for ii in range(num_v):
        if prizes[ii] > 0.0 and prizes[ii] < min_nonzero_prize:
            min_nonzero_prize = prizes[ii]
    eps = 0.01 * min_nonzero_prize

    if verbosity_level >= 1:
        print('Initial lambda_l = {}   lambda_r = {}   eps = {}'.format(lambda_l, lambda_r, eps))

    num_iter = 0
    while lambda_l - lambda_r > eps and num_iter < max_num_iter:
        num_iter += 1
        lambda_m = (lambda_l + lambda_r) / 2.0
        cur_vertices, cur_edges = pcst_fast.pcst_fast(opts.edges, prizes, lambda_m * costs, opts.root, opts.num_clusters, opts.pruning, verbosity_level - 1)
        cur_sparsity = cur_vertices.size
        if verbosity_level >= 1:
            print('lambda_l = {}   lambda_r = {}   lambda_m = {}   cur_sparsity = {}'.format(lambda_l, lambda_r, lambda_m, cur_sparsity))
        if cur_sparsity >= sparsity_low and cur_sparsity <= sparsity_high:
            if verbosity_level >= 1:
                print('Returning intermediate solution for lambda_m')
            result = np.zeros_like(prizes)
            result[cur_vertices] = prizes[cur_vertices]
            return result, cur_vertices, cur_edges
        if cur_sparsity > sparsity_high:
            lambda_r = lambda_m
        else:
            lambda_l = lambda_m
    cur_vertices, cur_edges = pcst_fast.pcst_fast(opts.edges, prizes, lambda_l * costs, opts.root, opts.num_clusters, opts.pruning, verbosity_level - 1)
    cur_sparsity = cur_vertices.size
    if cur_sparsity < sparsity_low:
        print('WARNING: returning sparsity {} although minimum sparsity {} was requested.'.format(cur_sparsity, sparsity_low))
    if verbosity_level >= 1:
        print('Returning final solution for lambda_l (cur_sparsity = {})'.format(cur_sparsity))
    result = np.zeros_like(prizes)
    result[cur_vertices] = prizes[cur_vertices]
    return result, cur_vertices, cur_edges



# Performs softmax regression with graph-sparsity constraints for the following
# inputs:
# X is the n x d data matrix (n examples in dimension d)
# y is the n-dimensional label vector.
#     Each entry is an integer between 0 and c-1 (inclusive), where c is the
#     number of classes.
# W0 is the c x d weight matrix
# sparsity_low and sparsity_high are the (approximate) upper and lower bounds
#     for the output sparsity
# graph_opts.edges, graph_opts.root, graph_opts.num_clusters, and
#     graph_opts.pruning are directly passed to graph_proj_sparsity (and hence
#     to pcst_fast)
# steps is the step size schedule, represented by a matrix of size
#     num_steps x num_choices. In each iteration, the algorithm tries all current
#     choices for the step size and chooses the one that makes largest progress.
# verbosity_level indicates whether intermediate output should be printed
#     verbosity_level - 1 is being passed to graph_proj_sparsity.
# graph_proj_max_num_iter is the maximum number of iterations in the
#     graph-sparsity projection.
# edge_costs is a real vector with non-negative edge costs
# edge_costs_multiplier: a factor weighing edge costs vs prizes
def gslr(X, y, W0, sparsity_low, sparsity_high, graph_opts, steps=None, verbosity_level=0, graph_proj_max_num_iter=20, edge_costs=None, edge_costs_multiplier=None):
    assert len(steps.shape) == 2
    num_steps, num_choices = steps.shape

    losses = np.zeros(num_steps + 1)
    losses[0] = softmax_loss(X, y, W0)
    W_cur = np.copy(W0)
    for ii in range(num_steps):
        print('iteration {}:'.format(ii + 1))
        grad = softmax_gradient(X, y, W_cur)
        best_loss = losses[ii]
        best_step_size = 0.0
        #print('initially best loss {}'.format(best_loss))
        for step_size in steps[ii,:]:
            W_tmp = W_cur - step_size * grad
            # print('before')
            # print(W_tmp)
            # print(softmax_loss(X, y, W_tmp))
            W_tmp = proj_softmax_matrix(W_tmp, sparsity_low, sparsity_high, graph_opts, verbosity_level - 1, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
            # print('after')
            # print(W_tmp)
            # print(softmax_loss(X, y, W_tmp))
            loss_next = softmax_loss(X, y, W_tmp)
            if loss_next < best_loss:
                best_loss = loss_next
                best_step_size = step_size
            if verbosity_level >= 1:
                print('  loss_cur = {}   loss_next = {}   step_size = {}'.format(losses[ii], loss_next, step_size))
        if verbosity_level >= 1:
            print('  best_step_size: {}'.format(best_step_size))
        W_cur -= best_step_size * grad
        W_cur = proj_softmax_matrix(W_cur, sparsity_low, sparsity_high, graph_opts, verbosity_level - 1, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
        losses[ii + 1] = softmax_loss(X, y, W_cur)
    return W_cur, losses

# Helper functions
def proj_softmax_matrix(W, sparsity_low, sparsity_high, opts, verbosity_level, graph_proj_max_num_iter, edge_costs=None, edge_costs_multiplier=None):
    c, d = W.shape
    W2 = np.square(W)
    for ii in range(c):
        W2[ii,:], _, _ = graph_proj_sparsity(W2[ii,:], sparsity_low, sparsity_high, opts, verbosity_level, graph_proj_max_num_iter, edge_costs, edge_costs_multiplier)
    return np.multiply(np.sign(W), np.sqrt(W2))

def predict(X, W):
    return np.argmax(np.dot(X, W.transpose()), axis=1)

def num_correct(y1, y2):
    return np.count_nonzero(np.equal(y1, y2))
