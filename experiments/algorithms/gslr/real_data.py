
# coding: utf-8

# # Graph-Sparse Logistic Regression applied to the real proteomics data from the TCGA/CPTAC Ovarian Cancer dataset.

import sys
sys.settrace
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import LabelEncoder

from matplotlib_venn import venn3, venn3_circles, venn2

repo_path = '/Users/alex/Documents/gslr/'
interactome_path = repo_path + 'experiments/algorithms/pcsf/inbiomap_temp.tsv'

sys.path.append(repo_path + 'gslr/')
import gslr

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


### V. Graph-Sparse Logistic Regression

d = len(nodes)
c = 2

graph_opts = gslr.GraphOptions(edges=edges, root=-1, num_clusters=1, pruning='strong')

sparsity_low = 150
sparsity_high = 350

verbosity_level = 1

num_steps = 25
possible_steps = np.array([0.03, 0.1, 0.3])
steps = np.tile(possible_steps, (num_steps, 1))

W0 = np.zeros((c, d))


W_hat, losses = gslr.gslr(X, y, W0, sparsity_low, sparsity_high, graph_opts, steps, verbosity_level, edge_costs=inbiomap_experimentally.cost.values, edge_costs_multiplier=6)
















