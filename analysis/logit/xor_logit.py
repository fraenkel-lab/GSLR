#!/usr/bin/env python

#SBATCH --partition sched_mem1TB_centos7
#SBATCH --job-name=LOGIT
#SBATCH --output=/home/lenail/proteomics/synthetic_proteomics/analysis/multiprocess_%j.out
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

	files = [("../generated_data/xor_ludwig_svd_normals/hsa00010_hsa00760_inbiomap_exp.csv", "hsa00010", "hsa00760"), ("../generated_data/xor_ludwig_svd_normals/hsa01230_hsa05222_inbiomap_exp.csv", "hsa01230", "hsa05222"), ("../generated_data/xor_ludwig_svd_normals/hsa04723_hsa05120_inbiomap_exp.csv", "hsa04723", "hsa05120"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00053_hsa00500_inbiomap_exp.csv", "hsa00053", "hsa00500"), ("../generated_data/xor_ludwig_svd_normals/hsa02010_hsa00300_inbiomap_exp.csv", "hsa02010", "hsa00300"), ("../generated_data/xor_ludwig_svd_normals/hsa04724_hsa04931_inbiomap_exp.csv", "hsa04724", "hsa04931"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00061_hsa05216_inbiomap_exp.csv", "hsa00061", "hsa05216"), ("../generated_data/xor_ludwig_svd_normals/hsa03010_hsa03013_inbiomap_exp.csv", "hsa03010", "hsa03013"), ("../generated_data/xor_ludwig_svd_normals/hsa04725_hsa03410_inbiomap_exp.csv", "hsa04725", "hsa03410"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00062_hsa04392_inbiomap_exp.csv", "hsa00062", "hsa04392"), ("../generated_data/xor_ludwig_svd_normals/hsa03015_hsa05322_inbiomap_exp.csv", "hsa03015", "hsa05322"), ("../generated_data/xor_ludwig_svd_normals/hsa04727_hsa05162_inbiomap_exp.csv", "hsa04727", "hsa05162"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00100_hsa03022_inbiomap_exp.csv", "hsa00100", "hsa03022"), ("../generated_data/xor_ludwig_svd_normals/hsa03018_hsa00030_inbiomap_exp.csv", "hsa03018", "hsa00030"), ("../generated_data/xor_ludwig_svd_normals/hsa04742_hsa04964_inbiomap_exp.csv", "hsa04742", "hsa04964"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00120_hsa04530_inbiomap_exp.csv", "hsa00120", "hsa04530"), ("../generated_data/xor_ludwig_svd_normals/hsa03020_hsa01524_inbiomap_exp.csv", "hsa03020", "hsa01524"), ("../generated_data/xor_ludwig_svd_normals/hsa04750_hsa03320_inbiomap_exp.csv", "hsa04750", "hsa03320"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00130_hsa00280_inbiomap_exp.csv", "hsa00130", "hsa00280"), ("../generated_data/xor_ludwig_svd_normals/hsa03030_hsa00250_inbiomap_exp.csv", "hsa03030", "hsa00250"), ("../generated_data/xor_ludwig_svd_normals/hsa04910_hsa05012_inbiomap_exp.csv", "hsa04910", "hsa05012"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00140_hsa00564_inbiomap_exp.csv", "hsa00140", "hsa00564"), ("../generated_data/xor_ludwig_svd_normals/hsa03040_hsa00400_inbiomap_exp.csv", "hsa03040", "hsa00400"), ("../generated_data/xor_ludwig_svd_normals/hsa04913_hsa04911_inbiomap_exp.csv", "hsa04913", "hsa04911"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00190_hsa04213_inbiomap_exp.csv", "hsa00190", "hsa04213"), ("../generated_data/xor_ludwig_svd_normals/hsa03050_hsa05412_inbiomap_exp.csv", "hsa03050", "hsa05412"), ("../generated_data/xor_ludwig_svd_normals/hsa04914_hsa04722_inbiomap_exp.csv", "hsa04914", "hsa04722"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00220_hsa05200_inbiomap_exp.csv", "hsa00220", "hsa05200"), ("../generated_data/xor_ludwig_svd_normals/hsa03060_hsa00071_inbiomap_exp.csv", "hsa03060", "hsa00071"), ("../generated_data/xor_ludwig_svd_normals/hsa04918_hsa00830_inbiomap_exp.csv", "hsa04918", "hsa00830"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00230_hsa04930_inbiomap_exp.csv", "hsa00230", "hsa04930"), ("../generated_data/xor_ludwig_svd_normals/hsa03420_hsa04072_inbiomap_exp.csv", "hsa03420", "hsa04072"), ("../generated_data/xor_ludwig_svd_normals/hsa04920_hsa04210_inbiomap_exp.csv", "hsa04920", "hsa04210"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00232_hsa04730_inbiomap_exp.csv", "hsa00232", "hsa04730"), ("../generated_data/xor_ludwig_svd_normals/hsa03440_hsa04010_inbiomap_exp.csv", "hsa03440", "hsa04010"), ("../generated_data/xor_ludwig_svd_normals/hsa04922_hsa01521_inbiomap_exp.csv", "hsa04922", "hsa01521"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00240_hsa04976_inbiomap_exp.csv", "hsa00240", "hsa04976"), ("../generated_data/xor_ludwig_svd_normals/hsa03450_hsa04974_inbiomap_exp.csv", "hsa03450", "hsa04974"), ("../generated_data/xor_ludwig_svd_normals/hsa04923_hsa04915_inbiomap_exp.csv", "hsa04923", "hsa04915"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00260_hsa00900_inbiomap_exp.csv", "hsa00260", "hsa00900"), ("../generated_data/xor_ludwig_svd_normals/hsa04012_hsa03430_inbiomap_exp.csv", "hsa04012", "hsa03430"), ("../generated_data/xor_ludwig_svd_normals/hsa04925_hsa00072_inbiomap_exp.csv", "hsa04925", "hsa00072"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00270_hsa00590_inbiomap_exp.csv", "hsa00270", "hsa00590"), ("../generated_data/xor_ludwig_svd_normals/hsa04014_hsa05214_inbiomap_exp.csv", "hsa04014", "hsa05214"), ("../generated_data/xor_ludwig_svd_normals/hsa04932_hsa04720_inbiomap_exp.csv", "hsa04932", "hsa04720"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00310_hsa04950_inbiomap_exp.csv", "hsa00310", "hsa04950"), ("../generated_data/xor_ludwig_svd_normals/hsa04015_hsa04216_inbiomap_exp.csv", "hsa04015", "hsa04216"), ("../generated_data/xor_ludwig_svd_normals/hsa04933_hsa04610_inbiomap_exp.csv", "hsa04933", "hsa04610"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00330_hsa05031_inbiomap_exp.csv", "hsa00330", "hsa05031"), ("../generated_data/xor_ludwig_svd_normals/hsa04020_hsa04640_inbiomap_exp.csv", "hsa04020", "hsa04640"), ("../generated_data/xor_ludwig_svd_normals/hsa04940_hsa05410_inbiomap_exp.csv", "hsa04940", "hsa05410"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00350_hsa04977_inbiomap_exp.csv", "hsa00350", "hsa04977"), ("../generated_data/xor_ludwig_svd_normals/hsa04022_hsa04064_inbiomap_exp.csv", "hsa04022", "hsa04064"), ("../generated_data/xor_ludwig_svd_normals/hsa04970_hsa04726_inbiomap_exp.csv", "hsa04970", "hsa04726"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00380_hsa04120_inbiomap_exp.csv", "hsa00380", "hsa04120"), ("../generated_data/xor_ludwig_svd_normals/hsa04060_hsa05223_inbiomap_exp.csv", "hsa04060", "hsa05223"), ("../generated_data/xor_ludwig_svd_normals/hsa04971_hsa04151_inbiomap_exp.csv", "hsa04971", "hsa04151"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00410_hsa04917_inbiomap_exp.csv", "hsa00410", "hsa04917"), ("../generated_data/xor_ludwig_svd_normals/hsa04115_hsa04916_inbiomap_exp.csv", "hsa04115", "hsa04916"), ("../generated_data/xor_ludwig_svd_normals/hsa04972_hsa05217_inbiomap_exp.csv", "hsa04972", "hsa05217"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00430_hsa00360_inbiomap_exp.csv", "hsa00430", "hsa00360"), ("../generated_data/xor_ludwig_svd_normals/hsa04122_hsa04919_inbiomap_exp.csv", "hsa04122", "hsa04919"), ("../generated_data/xor_ludwig_svd_normals/hsa05010_hsa04921_inbiomap_exp.csv", "hsa05010", "hsa04921"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00440_hsa04071_inbiomap_exp.csv", "hsa00440", "hsa04071"), ("../generated_data/xor_ludwig_svd_normals/hsa04130_hsa04144_inbiomap_exp.csv", "hsa04130", "hsa04144"), ("../generated_data/xor_ludwig_svd_normals/hsa05030_hsa00670_inbiomap_exp.csv", "hsa05030", "hsa00670"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00450_hsa05218_inbiomap_exp.csv", "hsa00450", "hsa05218"), ("../generated_data/xor_ludwig_svd_normals/hsa04137_hsa01212_inbiomap_exp.csv", "hsa04137", "hsa01212"), ("../generated_data/xor_ludwig_svd_normals/hsa05033_hsa04136_inbiomap_exp.csv", "hsa05033", "hsa04136"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00471_hsa00740_inbiomap_exp.csv", "hsa00471", "hsa00740"), ("../generated_data/xor_ludwig_svd_normals/hsa04140_hsa04510_inbiomap_exp.csv", "hsa04140", "hsa04510"), ("../generated_data/xor_ludwig_svd_normals/hsa05034_hsa04912_inbiomap_exp.csv", "hsa05034", "hsa04912"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00480_hsa05330_inbiomap_exp.csv", "hsa00480", "hsa05330"), ("../generated_data/xor_ludwig_svd_normals/hsa04141_hsa00970_inbiomap_exp.csv", "hsa04141", "hsa00970"), ("../generated_data/xor_ludwig_svd_normals/hsa05110_hsa04614_inbiomap_exp.csv", "hsa05110", "hsa04614"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00511_hsa04146_inbiomap_exp.csv", "hsa00511", "hsa04146"), ("../generated_data/xor_ludwig_svd_normals/hsa04145_hsa05164_inbiomap_exp.csv", "hsa04145", "hsa05164"), ("../generated_data/xor_ludwig_svd_normals/hsa05130_hsa00604_inbiomap_exp.csv", "hsa05130", "hsa00604"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00515_hsa00040_inbiomap_exp.csv", "hsa00515", "hsa00040"), ("../generated_data/xor_ludwig_svd_normals/hsa04152_hsa01210_inbiomap_exp.csv", "hsa04152", "hsa01210"), ("../generated_data/xor_ludwig_svd_normals/hsa05131_hsa05205_inbiomap_exp.csv", "hsa05131", "hsa05205"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00520_hsa00051_inbiomap_exp.csv", "hsa00520", "hsa00051"), ("../generated_data/xor_ludwig_svd_normals/hsa04211_hsa05150_inbiomap_exp.csv", "hsa04211", "hsa05150"), ("../generated_data/xor_ludwig_svd_normals/hsa05140_hsa05206_inbiomap_exp.csv", "hsa05140", "hsa05206"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00524_hsa05418_inbiomap_exp.csv", "hsa00524", "hsa05418"), ("../generated_data/xor_ludwig_svd_normals/hsa04260_hsa05016_inbiomap_exp.csv", "hsa04260", "hsa05016"), ("../generated_data/xor_ludwig_svd_normals/hsa05143_hsa04066_inbiomap_exp.csv", "hsa05143", "hsa04066"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00532_hsa04978_inbiomap_exp.csv", "hsa00532", "hsa04978"), ("../generated_data/xor_ludwig_svd_normals/hsa04261_hsa04142_inbiomap_exp.csv", "hsa04261", "hsa04142"), ("../generated_data/xor_ludwig_svd_normals/hsa05145_hsa04973_inbiomap_exp.csv", "hsa05145", "hsa04973"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00533_hsa04024_inbiomap_exp.csv", "hsa00533", "hsa04024"), ("../generated_data/xor_ludwig_svd_normals/hsa04320_hsa00020_inbiomap_exp.csv", "hsa04320", "hsa00020"), ("../generated_data/xor_ludwig_svd_normals/hsa05152_hsa05020_inbiomap_exp.csv", "hsa05152", "hsa05020"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00534_hsa04961_inbiomap_exp.csv", "hsa00534", "hsa04961"), ("../generated_data/xor_ludwig_svd_normals/hsa04340_hsa05310_inbiomap_exp.csv", "hsa04340", "hsa05310"), ("../generated_data/xor_ludwig_svd_normals/hsa05161_hsa00472_inbiomap_exp.csv", "hsa05161", "hsa00472"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00561_hsa00290_inbiomap_exp.csv", "hsa00561", "hsa00290"), ("../generated_data/xor_ludwig_svd_normals/hsa04350_hsa04540_inbiomap_exp.csv", "hsa04350", "hsa04540"), ("../generated_data/xor_ludwig_svd_normals/hsa05166_hsa04728_inbiomap_exp.csv", "hsa05166", "hsa04728"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00563_hsa04330_inbiomap_exp.csv", "hsa00563", "hsa04330"), ("../generated_data/xor_ludwig_svd_normals/hsa04370_hsa04810_inbiomap_exp.csv", "hsa04370", "hsa04810"), ("../generated_data/xor_ludwig_svd_normals/hsa05168_hsa04150_inbiomap_exp.csv", "hsa05168", "hsa04150"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00591_hsa00340_inbiomap_exp.csv", "hsa00591", "hsa00340"), ("../generated_data/xor_ludwig_svd_normals/hsa04371_hsa04110_inbiomap_exp.csv", "hsa04371", "hsa04110"), ("../generated_data/xor_ludwig_svd_normals/hsa05169_hsa00562_inbiomap_exp.csv", "hsa05169", "hsa00562"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00600_hsa00512_inbiomap_exp.csv", "hsa00600", "hsa00512"), ("../generated_data/xor_ludwig_svd_normals/hsa04512_hsa04520_inbiomap_exp.csv", "hsa04512", "hsa04520"), ("../generated_data/xor_ludwig_svd_normals/hsa05202_hsa04360_inbiomap_exp.csv", "hsa05202", "hsa04360"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00601_hsa01522_inbiomap_exp.csv", "hsa00601", "hsa01522"), ("../generated_data/xor_ludwig_svd_normals/hsa04514_hsa04924_inbiomap_exp.csv", "hsa04514", "hsa04924"), ("../generated_data/xor_ludwig_svd_normals/hsa05203_hsa04390_inbiomap_exp.csv", "hsa05203", "hsa04390"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00603_hsa05146_inbiomap_exp.csv", "hsa00603", "hsa05146"), ("../generated_data/xor_ludwig_svd_normals/hsa04550_hsa00052_inbiomap_exp.csv", "hsa04550", "hsa00052"), ("../generated_data/xor_ludwig_svd_normals/hsa05204_hsa04622_inbiomap_exp.csv", "hsa05204", "hsa04622"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00620_hsa01040_inbiomap_exp.csv", "hsa00620", "hsa01040"), ("../generated_data/xor_ludwig_svd_normals/hsa04611_hsa04962_inbiomap_exp.csv", "hsa04611", "hsa04962"), ("../generated_data/xor_ludwig_svd_normals/hsa05210_hsa04662_inbiomap_exp.csv", "hsa05210", "hsa04662"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00630_hsa05134_inbiomap_exp.csv", "hsa00630", "hsa05134"), ("../generated_data/xor_ludwig_svd_normals/hsa04612_hsa05133_inbiomap_exp.csv", "hsa04612", "hsa05133"), ("../generated_data/xor_ludwig_svd_normals/hsa05211_hsa05100_inbiomap_exp.csv", "hsa05211", "hsa05100"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00640_hsa00785_inbiomap_exp.csv", "hsa00640", "hsa00785"), ("../generated_data/xor_ludwig_svd_normals/hsa04620_hsa04380_inbiomap_exp.csv", "hsa04620", "hsa04380"), ("../generated_data/xor_ludwig_svd_normals/hsa05212_hsa05132_inbiomap_exp.csv", "hsa05212", "hsa05132"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00730_hsa04966_inbiomap_exp.csv", "hsa00730", "hsa04966"), ("../generated_data/xor_ludwig_svd_normals/hsa04621_hsa03008_inbiomap_exp.csv", "hsa04621", "hsa03008"), ("../generated_data/xor_ludwig_svd_normals/hsa05215_hsa04740_inbiomap_exp.csv", "hsa05215", "hsa04740"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00750_hsa04310_inbiomap_exp.csv", "hsa00750", "hsa04310"), ("../generated_data/xor_ludwig_svd_normals/hsa04623_hsa05220_inbiomap_exp.csv", "hsa04623", "hsa05220"), ("../generated_data/xor_ludwig_svd_normals/hsa05219_hsa00920_inbiomap_exp.csv", "hsa05219", "hsa00920"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00770_hsa04062_inbiomap_exp.csv", "hsa00770", "hsa04062"), ("../generated_data/xor_ludwig_svd_normals/hsa04630_hsa05144_inbiomap_exp.csv", "hsa04630", "hsa05144"), ("../generated_data/xor_ludwig_svd_normals/hsa05221_hsa05160_inbiomap_exp.csv", "hsa05221", "hsa05160"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00780_hsa00565_inbiomap_exp.csv", "hsa00780", "hsa00565"), ("../generated_data/xor_ludwig_svd_normals/hsa04657_hsa04270_inbiomap_exp.csv", "hsa04657", "hsa04270"), ("../generated_data/xor_ludwig_svd_normals/hsa05224_hsa05213_inbiomap_exp.csv", "hsa05224", "hsa05213"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00790_hsa05032_inbiomap_exp.csv", "hsa00790", "hsa05032"), ("../generated_data/xor_ludwig_svd_normals/hsa04658_hsa00531_inbiomap_exp.csv", "hsa04658", "hsa00531"), ("../generated_data/xor_ludwig_svd_normals/hsa05230_hsa04650_inbiomap_exp.csv", "hsa05230", "hsa04650"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00860_hsa04960_inbiomap_exp.csv", "hsa00860", "hsa04960"), ("../generated_data/xor_ludwig_svd_normals/hsa04659_hsa04080_inbiomap_exp.csv", "hsa04659", "hsa04080"), ("../generated_data/xor_ludwig_svd_normals/hsa05231_hsa00650_inbiomap_exp.csv", "hsa05231", "hsa00650"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00910_hsa00510_inbiomap_exp.csv", "hsa00910", "hsa00510"), ("../generated_data/xor_ludwig_svd_normals/hsa04660_hsa04744_inbiomap_exp.csv", "hsa04660", "hsa04744"), ("../generated_data/xor_ludwig_svd_normals/hsa05321_hsa03460_inbiomap_exp.csv", "hsa05321", "hsa03460"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00980_hsa04670_inbiomap_exp.csv", "hsa00980", "hsa04670"), ("../generated_data/xor_ludwig_svd_normals/hsa04664_hsa04713_inbiomap_exp.csv", "hsa04664", "hsa04713"), ("../generated_data/xor_ludwig_svd_normals/hsa05323_hsa05340_inbiomap_exp.csv", "hsa05323", "hsa05340"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00982_hsa00592_inbiomap_exp.csv", "hsa00982", "hsa00592"), ("../generated_data/xor_ludwig_svd_normals/hsa04666_hsa04668_inbiomap_exp.csv", "hsa04666", "hsa04668"), ("../generated_data/xor_ludwig_svd_normals/hsa05332_hsa05142_inbiomap_exp.csv", "hsa05332", "hsa05142"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa00983_hsa05320_inbiomap_exp.csv", "hsa00983", "hsa05320"), ("../generated_data/xor_ludwig_svd_normals/hsa04672_hsa04068_inbiomap_exp.csv", "hsa04672", "hsa04068"), ("../generated_data/xor_ludwig_svd_normals/hsa05414_hsa04215_inbiomap_exp.csv", "hsa05414", "hsa04215"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa01100_hsa05014_inbiomap_exp.csv", "hsa01100", "hsa05014"), ("../generated_data/xor_ludwig_svd_normals/hsa04710_hsa00514_inbiomap_exp.csv", "hsa04710", "hsa00514"), ("../generated_data/xor_ludwig_svd_normals/hsa05416_hsa01523_inbiomap_exp.csv", "hsa05416", "hsa01523"),
			 ("../generated_data/xor_ludwig_svd_normals/hsa01200_hsa04975_inbiomap_exp.csv", "hsa01200", "hsa04975"), ("../generated_data/xor_ludwig_svd_normals/hsa04721_hsa04070_inbiomap_exp.csv", "hsa04721", "hsa04070")]


	pool = multiprocessing.Pool(n_cpus)
	results = pool.map(logit, files)

	# results = {pathway_id: (scores, features) for (first_pathway_id, second_pathway_id, scores, features) in results}

	pickle.dump(results, open('xor_logit_results.pickle', 'wb'))

	# pathways = pkl.load(open("KEGG_pathway_gene_lists.pkl", "rb"))




# inbiomap_exp = pd.read_csv('../../data_preparation/interactomes/InBioMap/inbiomap_exp.normalized.cleaned.tsv', sep='\t')
# interactome = nx.from_pandas_dataframe(inbiomap_exp, 'protein1', 'protein2', edge_attr=False)
# interactome
