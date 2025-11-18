import os
import tempfile
import anndata as ad

import muon
import numpy as np
# import requests
import scanpy as sc
import scvi
import seaborn as sns
import torch

import umap as UM
from scipy.stats import zscore
import scipy
from sklearn.decomposition import PCA

import multigrate as mtg

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd


class Multigrate_module:
    def __init__(self, model, method_name, modality1, modality2):
        self.model = model
        self.method_name = method_name
        self.modality1 = modality1
        self.modality2 = modality2

    def setup_data(self, mod1, mod2, mod1_field = None, mod2_field = None, batch_key = None):
        ''' 
        make sure mod1 is always rna
        '''
        adata = mtg.data.organize_multimodal_anndatas(
                adatas=[[mod1], [mod2]],  # a list of anndata objects per modality, RNA-seq always goes first
                layers=[[mod1_field], [mod2_field]],  # if need to use data from .layers, if None use .X
            )
        
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()

        rna_indices_end = mod1.shape[1]

        mtg.model.MultiVAE.setup_anndata(
                                            adata,
                                            categorical_covariate_keys=batch_key,
                                            rna_indices_end=rna_indices_end,
                                        )
        self.mod1_field = mod1_field
        self.mod2_field = mod2_field
        self.batch_key = batch_key
        return adata


    def predict(self, adata):
        self.model.impute(adata)
        pred1 = adata.obsm["imputed_modality_0"]
        pred2 = adata.obsm["imputed_modality_1"]

        
        return pred1, pred2
    
    def correlate(self, adata, Zscore = True):
        pred1, pred2 = self.predict(adata)
 
        input1 = adata[:, adata.var['modality']=="0"].X
        input2 = adata[:, adata.var['modality']=="1"].X

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
    
    def correlate_each_sample(self, adata, sample_labels = 'NA', Zscore = True):
        pred1, pred2 = self.predict(adata)
        input1 = adata[:, adata.var['modality']=="0"].X
        input2 = adata[:, adata.var['modality']=="1"].X
        if Zscore == True:
            pred1 = zscore(pred1)
            pred1[np.isnan(pred1)] = 0

            pred2 = zscore(pred2)
            pred2[np.isnan(pred2)] = 0

            input1 = zscore(input1)
            input1[np.isnan(input1)] = 0

            input2 = zscore(input2)
            input2[np.isnan(input2)] = 0


        half_point = adata.shape[0]

        catData1 = np.vstack((input1, 
                              pred1))

        corr1 = np.corrcoef(catData1)
        corr1 = corr1[:half_point,half_point:]

        df1 = pd.DataFrame(dict(corr = np.diag(corr1),
                               method = self.method_name,
                               modality = self.modality1,
                               cell_type = sample_labels)
                       )
        
        catData2 = np.vstack((input2, 
                              pred2))

        corr2 = np.corrcoef(catData2)
        corr2 = corr2[:half_point,half_point:]
        df2 = pd.DataFrame(dict(corr = np.diag(corr2),
                               method = self.method_name,
                               modality = self.modality2,
                               cell_type = sample_labels)
                       )

        return df1, df2
    
    def CramerVonMises(self, adata):
        pred1, pred2 = self.predict(adata)
        input1 = adata[:, adata.var['modality']=="0"].X
        input2 = adata[:, adata.var['modality']=="1"].X
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

        input1 = adata[:, adata.var['modality']=="0"].X
        input2 = adata[:, adata.var['modality']=="1"].X

        mod1_latent_space = self.model.module.encoder_0(torch.tensor(input1).type(torch.float32))
        mod2_latent_space = self.model.module.encoder_1(torch.tensor(input2).type(torch.float32))
        
        mod1_translated = self.model.module.decoder_0(mod2_latent_space)
        mod2_translated = self.model.module.decoder_1(mod1_latent_space)

        return mod1_translated.detach().numpy(), mod2_translated.detach().numpy()
    
    ''' 
    all plotting related things below
    '''
    def plot_sample_correlation(self, df):
        sns.kdeplot(df, x = 'corr', hue = 'cell_type', bw_adjust=0.5, fill = True, common_norm=False)


    def plot(self, adata, mod_index, Zscore = True):
        pred1, pred2 = self.predict(adata)
        input1 = adata[:, adata.var['modality']=="0"].X
        input2 = adata[:, adata.var['modality']=="1"].X

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



            


