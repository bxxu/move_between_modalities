import numpy as np
import h5py
import pandas as pd
import scanpy as sc
import anndata as ad

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import math

import scipy
import random
import statistics
import umap as UM
from scipy.stats import zscore
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import seaborn as sns
from IPython.display import clear_output
import muon
import scvi

import sys
sys.path.append('/michorlab/bingxian/multi_modal_unpaired_matching/')
import common_utils




class MultiVI_module:
    def __init__(self, model, method_name, modality1, modality2):
        self.model = model
        self.method_name = method_name
        self.modality1 = modality1
        self.modality2 = modality2

    def setup_data(self, mod1, mod2):
        mdata = muon.MuData({"rna": mod1, "atac":mod2})

        scvi.model.MULTIVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "atac_layer": "atac",
            },
        )

        return mdata
    
    def predict(self, adata):
        pred1 = self.model.get_normalized_expression(adata, return_numpy = True)
        pred2 = self.model.get_normalized_accessibility(adata, return_numpy = True)

        return pred1, pred2
    
    def correlate(self, adata, Zscore = True):
        pred1, pred2 = self.predict(adata)

        input1 = adata.mod['rna'].X
        input2 = adata.mod['atac'].X


        if Zscore == True:
            pred1 = zscore(pred1)
            pred1[np.isnan(pred1)] = 0

            pred2 = zscore(pred2)
            pred2[np.isnan(pred2)] = 0

            input1 = zscore(input1)
            input1[np.isnan(input1)] = 0

            input2 = zscore(input2)
            input2[np.isnan(input2)] = 0

        return np.corrcoef(input1.flatten(), pred1.flatten())[0,1], np.corrcoef(input2.flatten(), pred2.flatten())[0,1]
    
    def correlate_each_sample(self, adata, sample_labels = "NA", Zscore = True):

        pred1, pred2 = self.predict(adata)

        input1 = adata.mod['rna'].X
        input2 = adata.mod['atac'].X

        df1 = common_utils.correlate_each_sample(pred1, input1, sample_labels, Zscore)
        df2 = common_utils.correlate_each_sample(pred2, input2, sample_labels, Zscore)

        return df1, df2
    
    def CramerVonmises(self, adata):
        pred1, pred2 = self.predict(adata)

        input1 = adata.mod['rna'].X
        input2 = adata.mod['atac'].X

        out1 = []
        out2 = []

        for i in range(pred1.shape[1]):
            cvm = scipy.stats.cramervonmises_2samp(pred1[:,i], input1[:,i])
            out1.append(cvm.statistic)
        for i in range(pred2.shape[1]):
            cvm = scipy.stats.cramervonmises_2samp(pred2[:,i], input2[:,i])
            out2.append(cvm.statistic)

        return np.array(out1), np.array(out2)
    
    def generate(self, adata):
        adata_rna = adata.mod['rna'].copy()
        adata_atac = adata.mod['atac'].copy()

        adata_rna_test = adata_rna.copy()
        adata_rna_test.X = adata_rna.X * 0

        adata_atac_test = adata_atac.copy()
        adata_atac_test.X = adata_atac.X * 0

        mdata_rna0 = self.setup_data(adata_rna_test, adata_atac)
        mdata_atac0 = self.setup_data(adata_rna, adata_atac_test)

        mod1_translated, _ = self.predict(mdata_rna0)
        _, mod2_translated = self.predict(mdata_atac0)

        return mod1_translated, mod2_translated
    
    def plot_sample_correlation(self, df):
        sns.kdeplot(df, x = 'corr', hue = 'cell_type', bw_adjust=0.5, fill = True, common_norm=False)


    def plot(self, adata, mod_index, Zscore = True):
        pred1, pred2 = self.predict(adata)

        input1 = adata.mod['rna'].X
        input2 = adata.mod['atac'].X

        if Zscore == True:
            pred1 = zscore(pred1)
            pred1[np.isnan(pred1)] = 0

            pred2 = zscore(pred2)
            pred2[np.isnan(pred2)] = 0

            input1 = zscore(input1)
            input1[np.isnan(input1)] = 0

            input2 = zscore(input2)
            input2[np.isnan(input2)] = 0

        if mod_index == 1:
            plt.scatter(input1, pred1, s = 1, alpha = 0.1)
        if mod_index == 2:
            plt.scatter(input2, pred2, s = 1, alpha = 0.1)






    
