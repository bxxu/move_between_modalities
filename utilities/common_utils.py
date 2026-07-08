import numpy as np
import torch
import scipy
import pandas as pd
import random
import umap as UM
from scipy.stats import zscore
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix

import seaborn as sns



def make_fig_better(res = 300):
       plt.figure(dpi=res)  # Try 150–300 for crisp visuals
       plt.gcf().patch.set_alpha(0.0)
       plt.gca().patch.set_alpha(0.0) 


def setup_plot():
    plt.gcf().set_size_inches(5,2)

    sns.despine(top=True, right=True)

        # Make axis lines thicker
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

        # Increase font size of axis labels and ticks
    plt.xlabel('mse', fontsize=14)
    plt.ylabel('probability', fontsize=14)


    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


def CramerVonMises_2samp(reconstructed, observed, dim = 1, Zscore = False, Fix = False):
        if Fix == True:
             reconstructed[reconstructed < 0] = 0
             

        if Zscore == True:
             reconstructed = zscore(reconstructed)
             reconstructed[np.isnan(reconstructed)] = 0
             

             observed = zscore(observed)
             observed[np.isnan(observed)] = 0


        out = []
        if dim == 1:
            for i in range(observed.shape[1]):
                cvm = scipy.stats.cramervonmises_2samp(reconstructed[:,i], observed[:,i])
                out.append(cvm.statistic)
        if dim == 0:
              for i in range(observed.shape[0]):
                cvm = scipy.stats.cramervonmises_2samp(reconstructed[i,:], observed[i,:])
                out.append(cvm.statistic)
        return np.array(out)

def correlate_each_sample(reconstructed, observed, sample_labels = "NA", Zscore = True):
        if Zscore == True:
                observed = zscore(observed)
                observed[np.isnan(observed)] = 0
                reconstructed = zscore(reconstructed)
                reconstructed[np.isnan(reconstructed)] = 0

        half_point = reconstructed.shape[0]

        catData = np.vstack((observed, 
                             reconstructed))

        corr = np.corrcoef(catData)
        corr = corr[:half_point,half_point:]

        df = pd.DataFrame(dict(corr = np.diag(corr),
                               cell_type = sample_labels)
                       )

        return df


def Stats_df(reconstructed, observed, sample_labels = "NA", Zscore = True, Method = "NA", lib_norm = False, log1p = False):
        if lib_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)

        if Zscore == True:
               reconstructed = zscore(reconstructed)
               observed = zscore(observed)

               reconstructed[np.isnan(reconstructed)] = 0
               observed[np.isnan(observed)] = 0
        
        df = correlate_each_sample(reconstructed, observed, sample_labels, Zscore)
        df['cramer'] = CramerVonMises_2samp(reconstructed, observed, dim = 0, Zscore=Zscore)
        df['method'] = Method
        df['mse'] = ((reconstructed - observed)**2).mean(axis = 1)

        return df

def feature_Correlation(reconstructed, observed, Method = "NA", lib_norm = False, log1p = False):
        if lib_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)
        

        corr_obs = np.corrcoef(observed.T)
        corr_pred = np.corrcoef(reconstructed.T)

        corr_obs[np.isnan(corr_obs)] = 0
        corr_pred[np.isnan(corr_pred)] = 0

        corr_diff = corr_obs[~np.eye(corr_obs.shape[0], dtype=bool)] - corr_pred[~np.eye(corr_pred.shape[0], dtype=bool)]
        
        df = pd.DataFrame(dict(corr_diff = corr_diff,
                               method = Method))
        return df

def sample_Correlation(reconstructed, observed, Method = "NA", lib_norm = False, log1p = False):
        if lib_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)
        
        observed[np.isnan(observed)] = 0
        reconstructed[np.isnan(reconstructed)] = 0

        corr_obs = np.corrcoef(observed)
        corr_pred = np.corrcoef(reconstructed)

        corr_obs[np.isnan(corr_obs)] = 0
        corr_pred[np.isnan(corr_pred)] = 0

        corr_diff = corr_obs[~np.eye(corr_obs.shape[0], dtype=bool)] - corr_pred[~np.eye(corr_pred.shape[0], dtype=bool)]
        
        df = pd.DataFrame(dict(corr_diff = corr_diff,
                               method = Method))
        return df



def feature_Cramer(reconstructed, observed, Method = "NA", lib_norm = False, log1p = False):
        if lib_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)
        

        CvM = CramerVonMises_2samp(reconstructed, observed, dim = 1, Zscore = False, Fix = False)
        
        df = pd.DataFrame(dict(cramer = CvM,
                               method = Method))
        return df


        
     

def common_embedding(umap_here, pca_here, reconstructed, observed, library_size_norm = False, log1p = False, Zscore = False, pca_raw = False):
        if library_size_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)

        if Zscore == True:
               reconstructed = zscore(reconstructed)
               observed = zscore(observed)

               reconstructed[np.isnan(reconstructed)] = 0
               observed[np.isnan(observed)] = 0


        reconstructed[np.isnan(reconstructed)] = 0
        observed[np.isnan(observed)] = 0

        if pca_raw == False:
                cat_data = np.vstack((reconstructed, observed))
                l = umap_here.fit_transform(pca_here.fit_transform(cat_data))
        else: 
               pca_here.fit(observed)
               cat_data = np.vstack((reconstructed, observed))
               l = umap_here.fit_transform(pca_here.transform(cat_data))
        
        try:
                df = pd.DataFrame(dict(x = l[:,0],
                                y = l[:,1],
                                hue = np.concat((np.repeat("reconstructed", reconstructed.shape[0]),
                                        np.repeat("real", observed.shape[0]))).flatten(),
                                #      cell_order = np.row_stack([reconstructed.shape[0] + observed.shape[0]]).flatten() 
                                )
                                )
        except AttributeError:
               df = pd.DataFrame(dict(x = l[:,0],
                                y = l[:,1],
                                hue = np.concatenate((np.repeat("reconstructed", reconstructed.shape[0]),
                                        np.repeat("real", observed.shape[0]))).flatten(),
                                #      cell_order = np.row_stack([reconstructed.shape[0] + observed.shape[0]]).flatten() 
                                )
                                )
               
        return df

def common_embedding_3(umap_here, pca_here, reconstructed, observed, translated, library_size_norm = False, log1p = False, Zscore = False):
        if library_size_norm == True:
                observed = observed / observed.sum(axis = 1).reshape(observed.shape[0],1)
                observed[np.isnan(observed)] = 0

        if log1p == True:
              observed = np.log(observed * 10000 + 1)
              reconstructed = np.log(reconstructed * 10000 + 1)
              translated = np.log(translated * 10000 + 1)

        if Zscore == True:
               reconstructed = zscore(reconstructed)
               translated = zscore(translated)
               observed = zscore(observed)

               reconstructed[np.isnan(reconstructed)] = 0
               translated[np.isnan(translated)] = 0
               observed[np.isnan(observed)] = 0


        reconstructed[np.isnan(reconstructed)] = 0
        translated[np.isnan(translated)] = 0
        observed[np.isnan(observed)] = 0

        cat_data = np.vstack((reconstructed, translated, observed))
        l = umap_here.fit_transform(pca_here.fit_transform(cat_data))
        labels = np.concatenate([
        np.repeat("reconstructed", reconstructed.shape[0]),
        np.repeat("translated", translated.shape[0]),
        np.repeat("real", observed.shape[0])
            ])
        df = pd.DataFrame(dict(x = l[:,0],
                             y = l[:,1],
                             hue = labels,
                             cell_order = np.row_stack([np.arange(l.shape[0])]).flatten() 
                             )
                        )
        return df
        

def loadDF(dir, subsample = None):
        df = pd.DataFrame()
        for i in dir:
                print(i)
                new_df = pd.read_pickle(i)
                if subsample != None:
                       idx = random.sample(range(new_df.shape[0]), min(new_df.shape[0], subsample) - 1)
                       new_df = new_df.iloc[idx]
                df = pd.concat([df, new_df])

        # if subsample != None:
        # #        print(df.shape[0])
        #        idx = random.sample(range(df.shape[0]), min(df.shape[0], subsample) - 1)
        #        df = df.iloc[idx]
        return df

def loadsingleDF(dir, subsample = None):
        df = pd.read_pickle(dir)
        if subsample != None:
                idx = random.sample(range(df.shape[0]), min(df.shape[0], subsample) - 1)
                df = df.iloc[idx]

        # if subsample != None:
        # #        print(df.shape[0])
        #        idx = random.sample(range(df.shape[0]), min(df.shape[0], subsample) - 1)
        #        df = df.iloc[idx]
        return df

      
def evaluate_neighbor(data1, data2, n_neighbor, run_pca = False, pc_dim = 50, run_umap = False, lib_norm = False):
       tpr = np.zeros(len(n_neighbor))
       fpr = np.zeros(len(n_neighbor))
       counter = 0
       if lib_norm == True:
              # lib_norm is lib norm followed by library size normalization
              data2 =  data2 / data2.sum(axis = 1).reshape(data2.shape[0],1)
              data2[np.isnan(data2)] = 0
              data2 = np.log(10000 * data2 + 1)

       if run_pca == True:
              pca_low = PCA(n_components = pc_dim, random_state = 1)
              data1 = pca_low.fit_transform(data1)
              data2 = pca_low.fit_transform(data2)

       if run_umap == True:
              reducer2 = UM.UMAP(n_neighbors=10, 
                  metric = 'euclidean', 
                  repulsion_strength=.1, 
                  n_epochs=500, 
                  negative_sample_rate=5, 
                  min_dist = .5,
                  verbose = False,
                  random_state=1,
                  n_components = 2
                 )
              
              data1 = reducer2.fit_transform(data1)
              data2 = reducer2.fit_transform(data2)
              

       for i in n_neighbor:
                print(i)
                graph1 = kneighbors_graph(data1, n_neighbors = i)
                graph2 = kneighbors_graph(data2, n_neighbors = i)
                tn, fp, fn, tp = confusion_matrix(graph2.toarray().flatten(), graph1.toarray().flatten()).ravel()
                tpr[counter] = tp / (tp + fn)
                fpr[counter] = fp / (fp + tn)   
                counter += 1    
       return tpr, fpr, np.array(n_neighbor)

def evaluate_neighbor_all_pairs(ref1, ref2, pred1, pred2, run_pca = False, pc_dim = 50, run_umap = False, lib_norm = False, max_ratio = 0.2, n_pts = 10):
       ''' 
       tpr2 is typically atac, or protein, no need to library normalize
       '''
       nsamp = pred1.shape[0]
       n_neighbor = np.linspace(5, nsamp * max_ratio, n_pts).astype(int)
       tpr1, fpr1, Nneighbor = evaluate_neighbor(pred1, ref1, n_neighbor, run_pca = run_pca, lib_norm = lib_norm, pc_dim = pc_dim, run_umap = run_umap)
       tpr2, fpr2, Nneighbor = evaluate_neighbor(pred2, ref2, n_neighbor, run_pca = run_pca, lib_norm = False, pc_dim = pc_dim, run_umap = run_umap)
       tpr3, fpr3, _ = evaluate_neighbor(ref2, ref1, n_neighbor, run_pca = run_pca, lib_norm = lib_norm, pc_dim = pc_dim, run_umap = run_umap)
       tpr4, fpr4, _ = evaluate_neighbor(pred1, pred2, n_neighbor, run_pca = run_pca, lib_norm = False, pc_dim = pc_dim, run_umap = run_umap)

       df1 = pd.DataFrame(dict(tpr = tpr1, fpr = fpr1, nneighbor = Nneighbor,label = 'modality 1', ratio = Nneighbor / nsamp))
       df2 = pd.DataFrame(dict(tpr = tpr2, fpr = fpr2, nneighbor = Nneighbor,label = 'modality 2', ratio = Nneighbor / nsamp))
       df3 = pd.DataFrame(dict(tpr = tpr3, fpr = fpr3, nneighbor = Nneighbor,label = 'intermodality observed', ratio = Nneighbor / nsamp))
       df4 = pd.DataFrame(dict(tpr = tpr4, fpr = fpr4, nneighbor = Nneighbor,label = 'intermodality reconstructed', ratio = Nneighbor / nsamp))

       final_df = pd.concat([df1, df2, df3, df4])
       return final_df

def knn_dist(data1, data2, pca = True, pc_dim = 50):
        data = np.vstack((data1, data2))
        if pca == True:
               pca_low = PCA(n_components = pc_dim, random_state = 1)
               data = pca_low.fit_transform(data)

        n_k = 1
        graph = kneighbors_graph(data, n_neighbors = n_k)
        n_comp, _ = connected_components(graph, directed=False)
        n_k = n_k + 1
        while n_comp > 1:
                graph = kneighbors_graph(data, n_neighbors = n_k)
                n_comp, _ = connected_components(graph, directed=False)
                n_k += 1
        
        print('k for knn is:' + str(n_k - 1))
        geodesic_distances = shortest_path(csgraph=graph, directed=False)
        inner1 = geodesic_distances[0:data1.shape[0], 0:data1.shape[0]]
        inner2 = geodesic_distances[data1.shape[0]:, data1.shape[0]:]
        cross = geodesic_distances[0:data1.shape[0], data1.shape[0]:]

        df1 = pd.DataFrame(dict(D = inner1.flatten(),
                                label = 'in group reconstructed'))

        df2 = pd.DataFrame(dict(D = inner2.flatten(),
                                label = 'in group observed'))

        df3 = pd.DataFrame(dict(D = cross.flatten(),
                                label = 'out group'))

        all_df = pd.concat([df1, df2, df3])

        return all_df


def knn_dist_3(data1, data2, data3, pca = True, pc_dim = 50, library_size_norm = False, log1p = False, Zscore = False):
        ''' 
        make sure data1 is real
        data2 is reconstructed
        data3 is translated
        '''
        if library_size_norm == True:
                data1 = data1 / data1.sum(axis = 1).reshape(data1.shape[0],1)
                data1[np.isnan(data1)] = 0

        if log1p == True:
              data1 = np.log(data1 * 10000 + 1)
              data2 = np.log(data2 * 10000 + 1)
              data3 = np.log(data3 * 10000 + 1)

        if Zscore == True:
               data2 = zscore(data2)
               data3 = zscore(data3)
               data1 = zscore(data1)

               data2[np.isnan(data2)] = 0
               data3[np.isnan(data3)] = 0
               data1[np.isnan(data1)] = 0

        data = np.vstack((data1, data2, data3))
        
        if pca == True:
               pca_low = PCA(n_components = pc_dim, random_state = 1)
               data = pca_low.fit_transform(data)

        n_k = 1
        graph = kneighbors_graph(data, n_neighbors = n_k)
        n_comp, _ = connected_components(graph, directed=False)
        n_k = n_k + 1
        while n_comp > 1:
                graph = kneighbors_graph(data, n_neighbors = n_k)
                n_comp, _ = connected_components(graph, directed=False)
                n_k += 1
        
        print('k for knn is:' + str(n_k - 1))
        geodesic_distances = shortest_path(csgraph=graph, directed=False)
        ed2 = int(data1.shape[0] + data2.shape[0])
        
        inner1 = geodesic_distances[0:data1.shape[0], 0:data1.shape[0]] #real to real
        inner2 = geodesic_distances[data1.shape[0]:ed2, 0:data1.shape[0]] #reconst to real
        cross = geodesic_distances[ed2:, 0:data1.shape[0]] #translate to real

        print(inner1.shape)
        print(inner2.shape)
        print(cross.shape)

        df1 = pd.DataFrame(dict(D = inner1.flatten(),
                                label = 'real'))

        df2 = pd.DataFrame(dict(D = inner2.flatten(),
                                label = 'reconstructed'))

        df3 = pd.DataFrame(dict(D = cross.flatten(),
                                label = 'translated'))

        all_df = pd.concat([df1, df2, df3])

        return all_df



def graph_embedding_label_agreement(data, label, n_k=10):
       graph = kneighbors_graph(data, n_neighbors = n_k)
       ratio = []
       for i in range(graph.shape[0]):
                cell_type_here = label[i]

                use_idx = (graph[i,:].toarray() == 1).flatten()
                all_cell_types = label[use_idx]

                ratio.append((all_cell_types == cell_type_here).sum() / n_k)

       ratio = np.array(ratio)
       
       return ratio

       

def knn_dist_5(data1, data2, data3, data4, data5):
       

        data = np.vstack((data1, data2, data3, data4, data5))
        
       
        n_k = 1
        graph = kneighbors_graph(data, n_neighbors = n_k)
        n_comp, _ = connected_components(graph, directed=False)
        n_k = n_k + 1
        while n_comp > 1:
                graph = kneighbors_graph(data, n_neighbors = n_k)
                n_comp, _ = connected_components(graph, directed=False)
                n_k += 1
        
        print('k for knn is:' + str(n_k - 1))
        geodesic_distances = shortest_path(csgraph=graph, directed=False)

        n1, n2, n3, n4, n5 = data1.shape[0], data2.shape[0], data3.shape[0], data4.shape[0], data5.shape[0]
        s1 = slice(0, n1)
        s2 = slice(n1, n1 + n2)
        s3 = slice(n1 + n2, n1 + n2 + n3)
        s4 = slice(n1 + n2 + n3, n1 + n2 + n3 + n4)
        s5 = slice(n1 + n2 + n3 + n4, n1 + n2 + n3 + n4 + n5)

        inner = geodesic_distances[s1, s1] 
        cross1 = geodesic_distances[s2, s1] 
        cross2 = geodesic_distances[s3, s1] 
        cross3 = geodesic_distances[s4, s1]
        cross4 = geodesic_distances[s5,s1]

        df1 = pd.DataFrame(dict(D = inner.flatten(),
                                label = 'zygote'))

        df2 = pd.DataFrame(dict(D = cross1.flatten(),
                                label = '2cell'))

        df3 = pd.DataFrame(dict(D = cross2.flatten(),
                                label = '4cell'))
        
        df4 = pd.DataFrame(dict(D = cross3.flatten(),
                                label = '8cell'))

        df5 = pd.DataFrame(dict(D = cross4.flatten(),
                                label = 'E3.0'))
        

        all_df = pd.concat([df1, df2, df3, df4, df5])

        return all_df


def knn_graph1(data):
       n_k = 1
       graph = kneighbors_graph(data, n_neighbors = n_k)
       n_comp, _ = connected_components(graph, directed=False)
       n_k = n_k + 1
       while n_comp > 1:
                graph = kneighbors_graph(data, n_neighbors = n_k)
                n_comp, _ = connected_components(graph, directed=False)
                n_k += 1
       geodesic_distances = shortest_path(csgraph=graph, directed=False)

       return geodesic_distances

def train_model_benchmark(mod1, mod2, annot, use_method, file_path, test_set_size = 0.1, max_epoch_num = 500):
    import umap as UM
    from scipy.stats import zscore
    from sklearn.decomposition import PCA
    import os

    os.makedirs(file_path, exist_ok=True) 
    network_node = 200
    network_latent_dim  = 20


    pca_low = PCA(n_components = 50, random_state = 1)
    reducer2 = UM.UMAP(n_neighbors=10, 
                    metric = 'euclidean', 
                    repulsion_strength=.1, 
                    n_epochs=500, 
                    negative_sample_rate=5, 
                    min_dist = .5,
                    verbose = False,
                    random_state=1,
                    n_components = 2
                    )

    if (use_method == "scVI") | (use_method == "multiVI") | (use_method == 'peakVI'):
        import os
        print('limit threads')
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
        os.environ["NUMEXPR_NUM_THREADS"] = "2"


    print('start loading')

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import math
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import anndata as ad

    import matplotlib
    import matplotlib.pyplot as plt


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

    print('done loading everything')


    import sys
    sys.path.append('/michorlab/bingxian/multi_modal_unpaired_matching/')
    # from utils_MBTA import *
    import common_utils



    print(test_set_size)
    test_set_size = int(mod1.shape[0] * test_set_size)
    

    train_set_size = mod1.shape[0] - test_set_size


    mod1.X = mod1.X.toarray() if hasattr(mod1.X, "toarray") else mod1.X
    mod2.X = mod2.X.toarray() if hasattr(mod2.X, "toarray") else mod2.X  # binarize for atac

    # mod2.X = (mod2.X >0).astype(np.float32)

    print(mod1)
    print(mod2)

#     if (use_method != 'scVI') & ( use_method != 'multiVI'):
#         sc.pp.normalize_total(mod1, target_sum=1e4)
#         sc.pp.log1p(mod1)
    if use_method == 'MBTA':
        mod1_tensor = torch.tensor(mod1.X).type(torch.float32)
        mod2_tensor = torch.tensor(mod2.X).type(torch.float32)
        mod1_tensor_ori = mod1_tensor.clone()
        mod2_tensor_ori = mod2_tensor.clone()

        idx = random.sample(range(mod1_tensor.shape[0]), test_set_size)
        remaining_idx = np.setdiff1d(np.arange(mod1_tensor.shape[0]), idx)

        mod1_tensor_test = mod1_tensor[idx,:]
        mod1_tensor = mod1_tensor[remaining_idx,:]
        mod2_tensor_test = mod2_tensor[idx,:]
        mod2_tensor = mod2_tensor[remaining_idx,:]

        if test_set_size == 0:
            mod1_tensor = torch.tensor(mod1.X).type(torch.float32)
            mod2_tensor = torch.tensor(mod2.X).type(torch.float32)
            mod1_tensor_test = mod1_tensor.clone()
            mod2_tensor_test = mod2_tensor.clone()

    else:
        idx = random.sample(range(mod1.shape[0]), test_set_size)
        remaining_idx = np.setdiff1d(np.arange(mod1.shape[0]), idx)

        mod1_tensor_test = mod1[idx].copy()
        mod1_tensor = mod1[remaining_idx].copy()

        mod2_tensor_test = mod2[idx].copy()
        mod2_tensor = mod2[remaining_idx].copy()

        test_set_index = idx.copy()

        if test_set_size == 0:
            mod1_tensor = mod1.copy()
            mod2_tensor = mod2.copy()
            mod1_tensor_test = mod1.copy()
            mod2_tensor_test = mod2.copy()

    print('done loading data')

    if use_method == 'MBTA':
        import utils_MBTA

        use_iter = int(max_epoch_num * (mod1_tensor.shape[0] / 150)) # this does not exist before. 
        
        model_mod1 = utils_MBTA.train_model(mod1_tensor,
                                    Nnodes = network_node, #ori 200
                                    latent_dim = network_latent_dim, # ori is 20
                                    batchSize = 150,
                                    Ndir = 50, Niter = 20000) # ori is 20000

        print("done with RNA")
        if (mod2_tensor!=0).sum() / (mod2_tensor.shape[0] * mod2_tensor.shape[1]) < 0.01:
            use_balance = True
        else:
            use_balance = False

        print('am i balancing the zeros or no?')
        print(use_balance)


        model_mod2 = utils_MBTA.train_model(mod2_tensor,
                                        Nnodes = network_node, #ori 200
                                        latent_dim = network_latent_dim, # ori is 20
                                        batchSize = 150,
                                        Ndir = 50, Niter = 20000, balance = use_balance, neg_ratio = 10)
        
        m1 = utils_MBTA.MBTA_submodule(model_mod1, "MBTA", "RNA")
        m2 = utils_MBTA.MBTA_submodule(model_mod2, "MBTA", "ATAC")

        MBTA = utils_MBTA.paired_MBTA(m1, m2)
        MBTA.paired_training(mod1_tensor, mod2_tensor,  Nitr=50000)
        MBTA.paired_training(mod1_tensor, mod2_tensor,  Nitr=50000, Dir = 2)
        clear_output()

        pred1 = m1.predict(mod1_tensor_test)
        pred2 = m2.predict(mod2_tensor_test)
        a = MBTA.generate(m2.model.latentMean(mod2_tensor_test), m1.model.latentMean(mod1_tensor_test), Dir = 2)
        b = MBTA.generate(m1.model.latentMean(mod1_tensor_test), m2.model.latentMean(mod2_tensor_test))

        mod1_tensor_test = mod1_tensor_test.detach().numpy()
        mod2_tensor_test = mod2_tensor_test.detach().numpy()

        input = mod1_tensor_ori.clone()
        input = m1.model.latentMean(input)
        mvn = torch.distributions.MultivariateNormal(torch.zeros(m2.model.latentDim), 
                                                    torch.eye(m2.model.latentDim))

        inputX = mvn.sample([input.shape[0]])
        out = utils_MBTA.ode_forward(MBTA.MBTA_12, input, inputX, 0.01)
        out = out[-1,:,:]

        cell_embedding = reducer2.fit_transform(out.detach().numpy())
        project_modality1 = common_embedding_3(reducer2, pca_low, 
                                               m1.model.back2high(m1.model.latentMean(mod1_tensor_ori)).detach().numpy(),
                                               mod1_tensor_ori.detach().numpy(),
                                               MBTA.generate(m2.model.latentMean(mod2_tensor_ori), m1.model.latentMean(mod1_tensor_ori), Dir = 2))
        project_modality2 = common_embedding_3(reducer2, pca_low, 
                                               m2.model.back2high(m2.model.latentMean(mod2_tensor_ori)).detach().numpy(),
                                               mod2_tensor_ori.detach().numpy(),
                                               MBTA.generate(m1.model.latentMean(mod1_tensor_ori), m2.model.latentMean(mod2_tensor_ori)))

    if use_method == "multigrate":
        import utils_Multigrate
        import multigrate as mtg
        print('importing multigrate')

        random.seed(None)
        seed = random.randint(0, 2**32 - 1)
        from lightning import seed_everything
        seed_everything(seed)
        print(seed)

        adata = mtg.data.organize_multimodal_anndatas(
        adatas=[[mod1_tensor], [mod2_tensor]],  # a list of anndata objects per modality, RNA-seq always goes first
        layers=[[None], [None]],  # if need to use data from .layers, if None use .X
        )

        rna_indices_end = mod1_tensor.shape[1]
        mtg.model.MultiVAE.setup_anndata(
            adata,
            # categorical_covariate_keys=["Site"], # cannot have batch key or dimension do not match
            rna_indices_end=rna_indices_end,
        )

        vae = mtg.model.MultiVAE(
            adata,
            losses=["mse", "mse"],
            n_layers_encoders = [4,4],
            n_layers_decoders = [4,4],
            n_hidden_encoders = [network_node,network_node],
            n_hidden_decoders = [network_node,network_node],
            z_dim = network_latent_dim
        )

        vae.train(max_epochs=max_epoch_num, early_stopping=True)
        Mg = utils_Multigrate.Multigrate_module(vae, 'multigrate', "RNA", "ATAC")
        test = Mg.setup_data(mod1_tensor_test, mod2_tensor_test)
        pred1, pred2 = Mg.predict(test)
        a,b = Mg.generate(test)

        mod1_tensor_test = mod1_tensor_test.X
        mod2_tensor_test = mod2_tensor_test.X

        full_data = Mg.setup_data(mod1, mod2)
        vae.get_model_output(full_data)
        cell_embedding = reducer2.fit_transform(full_data.obsm['X_multigrate'])
        pred1_2, pred2_2 = Mg.predict(full_data)
        a2,b2 = Mg.generate(full_data)

        project_modality1 = common_embedding_3(reducer2, pca_low, 
                                               pred1_2,
                                               mod1.X,
                                               a2)
        project_modality2 = common_embedding_3(reducer2, pca_low, 
                                               pred2_2,
                                               mod2.X,
                                               b2)
    if use_method == "multiVI":
        import utils_MultiVI
        import scvi
        

        mdata = muon.MuData({"rna": mod1_tensor, "atac":mod2_tensor})

        scvi.model.MULTIVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "atac_layer": "atac",
            },
        )

        model = scvi.model.MULTIVI(
            mdata,
            n_genes=len(mdata.mod["rna"].var),
            n_regions=len(mdata.mod["atac"].var),
            n_hidden = network_node,
            n_latent = network_latent_dim,
            n_layers_encoder=4, 
            n_layers_decoder=4,
        )

        print(mdata.shape)
        model.train(max_epochs = max_epoch_num)
        Mv = utils_MultiVI.MultiVI_module(model, "multiVI", "RNA", "ATAC")
        test = Mv.setup_data(mod1_tensor_test, mod2_tensor_test)
        pred1, pred2 = Mv.predict(test)
        a,b = Mv.generate(test)

        mod1_tensor_test = mod1_tensor_test.X
        mod2_tensor_test = mod2_tensor_test.X

        full_data = Mv.setup_data(mod1, mod2)
        full_data.obsm['X_multivi'] = model.get_latent_representation(full_data)
        cell_embedding = reducer2.fit_transform(full_data.obsm['X_multivi'])
        pred1_2, pred2_2 = Mv.predict(full_data)
        a2,b2 = Mv.generate(full_data)

        project_modality1 = common_embedding_3(reducer2, pca_low, 
                                               pred1_2,
                                               mod1.X,
                                               a2,
                                               library_size_norm=True,
                                               log1p = True)
        project_modality2 = common_embedding_3(reducer2, pca_low, 
                                               pred2_2,
                                               mod2.X,
                                               b2,
                                               library_size_norm=True,
                                               log1p = True)
    if use_method == "peakVI":
        import scvi
        scvi.model.PEAKVI.setup_anndata(mod2_tensor)

        model = scvi.model.PEAKVI(
            mod2_tensor,
            n_hidden = network_node,
            n_latent = network_latent_dim,
            n_layers_encoder = 4,
            n_layers_decoder = 4
        )

        model.train(max_epochs = max_epoch_num)
        scvi.model.PEAKVI.setup_anndata(mod2_tensor_test)
        pred2 =model.get_normalized_accessibility(mod2_tensor_test, return_numpy = True)

        mod1_tensor_test = mod1_tensor_test.X
        mod2_tensor_test = mod2_tensor_test.X
    if use_method == "scVI":
        import scvi
        scvi.model.SCVI.setup_anndata(mod1_tensor, layer=None)

        model = scvi.model.SCVI(
            mod1_tensor,
            n_hidden = network_node,
            n_latent = network_latent_dim,
            n_layers = 4,
        )

        print(mod1_tensor.shape)
        model.train(max_epochs = max_epoch_num)
        scvi.model.PEAKVI.setup_anndata(mod1_tensor_test)

        pred1 =model.get_normalized_expression(mod1_tensor_test, return_numpy = True)

        mod1_tensor_test = mod1_tensor_test.X
        mod2_tensor_test = mod2_tensor_test.X

    print('done training')

    Z = False #our approach works better when Z=false it seems sad

    path = file_path + str(train_set_size) + "_"
    method = str(network_node) + "_" + str(network_latent_dim) + "_" + use_method

    if (use_method != "scVI") & (use_method != "peakVI"):

        if (use_method == 'multiVI'):
            df1 =  Stats_df(pred1, mod1_tensor_test, Zscore=Z, Method=use_method, lib_norm = True, log1p=True)
            df2 =  Stats_df(pred2, mod2_tensor_test, Zscore=Z, Method=use_method)
            df3 =  Stats_df(a, mod1_tensor_test, Zscore=Z, Method=use_method, lib_norm = True, log1p=True)
            df4 =  Stats_df(b, mod2_tensor_test, Zscore=Z, Method=use_method)

            df5 =  feature_Correlation(pred1, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)
            df7 =  feature_Correlation(a, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)

            df9 =  sample_Correlation(pred1, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)
            df10 =  sample_Correlation(pred2, mod2_tensor_test, Method=use_method)
            df11 =  sample_Correlation(a, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)
            df12 =  sample_Correlation(b, mod2_tensor_test, Method=use_method)

            df13 =  feature_Cramer(pred1, mod1_tensor_test, Method = use_method, lib_norm = True, log1p=True)
            df14 =  feature_Cramer(pred2, mod2_tensor_test, Method = use_method)
            df15 =  feature_Cramer(a, mod1_tensor_test, Method = use_method, lib_norm = True, log1p=True)
            df16 =  feature_Cramer(b, mod2_tensor_test, Method = use_method)

            df17 =  evaluate_neighbor_all_pairs(mod1_tensor_test, mod2_tensor_test, pred1, pred2, run_pca = True, lib_norm=True)
            df18 =  evaluate_neighbor_all_pairs(mod1_tensor_test, mod2_tensor_test, a, b, run_pca = True, lib_norm=True)
            
            df19 = pd.DataFrame(dict(x = cell_embedding[:,0], y = cell_embedding[:,1], annot = annot))
            df20 = project_modality1
            df21 = project_modality2

            dfs = [df1, df2, df3, df4, df5, df7, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21]
            for i in dfs:
                i['n_hidden'] = network_node
                i['n_latent'] = network_latent_dim
                i['test_set'] = test_set_size
                i['train_set'] = train_set_size

            df1.to_pickle(path + method + '_df1.pkl')
            df2.to_pickle(path + method + '_df2.pkl')
            df3.to_pickle(path + method + '_df3.pkl')
            df4.to_pickle(path + method + '_df4.pkl')

            df5.to_pickle(path + method + '_df5.pkl')
            df7.to_pickle(path + method + '_df7.pkl')

            df9.to_pickle(path + method + '_df9.pkl')
            df10.to_pickle(path + method + '_df10.pkl')
            df11.to_pickle(path + method + '_df11.pkl')
            df12.to_pickle(path + method + '_df12.pkl')

            df13.to_pickle(path + method + '_df13.pkl')
            df14.to_pickle(path + method + '_df14.pkl')
            df15.to_pickle(path + method + '_df15.pkl')
            df16.to_pickle(path + method + '_df16.pkl')
            df17.to_pickle(path + method + '_df17.pkl')
            df18.to_pickle(path + method + '_df18.pkl')

            df19.to_pickle(path + method + '_df19.pkl')
            df20.to_pickle(path + method + '_df20.pkl')
            df21.to_pickle(path + method + '_df21.pkl')


        if (use_method == 'multigrate') | (use_method == 'MBTA'): 
            df1 = Stats_df(pred1, mod1_tensor_test, Zscore=Z, Method=use_method)
            df2 = Stats_df(pred2, mod2_tensor_test, Zscore=Z, Method=use_method)
            df3 = Stats_df(a, mod1_tensor_test, Zscore=Z, Method=use_method)
            df4 = Stats_df(b, mod2_tensor_test, Zscore=Z, Method=use_method)

            df5 =  feature_Correlation(pred1, mod1_tensor_test, Method=use_method)
            df7 =  feature_Correlation(a, mod1_tensor_test, Method=use_method)

            df9 =  sample_Correlation(pred1, mod1_tensor_test, Method=use_method)
            df10 =  sample_Correlation(pred2, mod2_tensor_test, Method=use_method)
            df11 =  sample_Correlation(a, mod1_tensor_test, Method=use_method)
            df12 =  sample_Correlation(b, mod2_tensor_test, Method=use_method)

            df13 =  feature_Cramer(pred1, mod1_tensor_test, Method = use_method)
            df14 =  feature_Cramer(pred2, mod2_tensor_test, Method = use_method)
            df15 =  feature_Cramer(a, mod1_tensor_test, Method = use_method)
            df16 =  feature_Cramer(b, mod2_tensor_test, Method = use_method)

            df17 =  evaluate_neighbor_all_pairs(mod1_tensor_test, mod2_tensor_test, pred1, pred2, run_pca = True)
            df18 =  evaluate_neighbor_all_pairs(mod1_tensor_test, mod2_tensor_test, a, b, run_pca = True)
            
            df19 = pd.DataFrame(dict(x = cell_embedding[:,0], y = cell_embedding[:,1], annot = annot))
            df20 = project_modality1
            df21 = project_modality2

            dfs = [df1, df2, df3, df4, df5, df7, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21]
            for i in dfs:
                i['n_hidden'] = network_node
                i['n_latent'] = network_latent_dim
                i['test_set'] = test_set_size
                i['train_set'] = train_set_size

            
            df1.to_pickle(path + method + '_df1.pkl')
            df2.to_pickle(path + method + '_df2.pkl')
            df3.to_pickle(path + method + '_df3.pkl')
            df4.to_pickle(path + method + '_df4.pkl')

            df5.to_pickle(path + method + '_df5.pkl')
            df7.to_pickle(path + method + '_df7.pkl')

            df9.to_pickle(path + method + '_df9.pkl')
            df10.to_pickle(path + method + '_df10.pkl')
            df11.to_pickle(path + method + '_df11.pkl')
            df12.to_pickle(path + method + '_df12.pkl')

            df13.to_pickle(path + method + '_df13.pkl')
            df14.to_pickle(path + method + '_df14.pkl')
            df15.to_pickle(path + method + '_df15.pkl')
            df16.to_pickle(path + method + '_df16.pkl')
            df17.to_pickle(path + method + '_df17.pkl')
            df18.to_pickle(path + method + '_df18.pkl')

            df19.to_pickle(path + method + '_df19.pkl')
            df20.to_pickle(path + method + '_df20.pkl')
            df21.to_pickle(path + method + '_df21.pkl')

    if use_method == "scVI":
        df1 =  Stats_df(pred1, mod1_tensor_test, Zscore=Z, Method=use_method, lib_norm = True, log1p=True)
        df5 =  feature_Correlation(pred1, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)
        df9 =  sample_Correlation(pred1, mod1_tensor_test, Method=use_method, lib_norm = True, log1p=True)

        df13 =  feature_Cramer(pred1, mod1_tensor_test, Method = use_method, lib_norm = True, log1p=True)
    
        dfs = [df1, df5, df9, df13]
        for i in dfs:
            i['n_hidden'] = network_node
            i['n_latent'] = network_latent_dim
            i['test_set'] = test_set_size
            i['train_set'] = train_set_size

        df1.to_pickle(path + method + '_df1.pkl')
        df5.to_pickle(path + method + '_df5.pkl')
        df9.to_pickle(path + method + '_df9.pkl')
        df13.to_pickle(path + method + '_df13.pkl')

    if use_method == "peakVI":
        df2 =  Stats_df(pred2, mod2_tensor_test, Zscore=Z, Method=use_method)
        df10 =  sample_Correlation(pred2, mod2_tensor_test, Method=use_method)

        df14 =  feature_Cramer(pred2, mod2_tensor_test, Method = use_method)

        dfs = [df2, df10, df14]
        for i in dfs:
            i['n_hidden'] = network_node
            i['n_latent'] = network_latent_dim
            i['test_set'] = test_set_size
            i['train_set'] = train_set_size

        df2.to_pickle(path + method + '_df2.pkl')
        df10.to_pickle(path + method + '_df10.pkl')
        df14.to_pickle(path + method + '_df14.pkl')
       

       

              
       
