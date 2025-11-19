'''
ues this to shorten what mbta needs
'''
import numpy as np
import h5py
# import hdf5plugin
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

'''
section for flow matching
'''
def sample_wasserstein(distribution, sample, dirNum):
    Ndim_samp = sample.shape[1]
    Nsamp = sample.shape[0]
    
    sampled_directions = torch.randn(dirNum, Ndim_samp)
    sampled_directions = sampled_directions / torch.linalg.norm(sampled_directions, dim = 1).view(sampled_directions.shape[0],1)
    dist_samples = distribution.sample([Nsamp])
    
    dist_projected = torch.matmul(dist_samples, sampled_directions.t())
    test_projected = torch.matmul(sample, sampled_directions.t())

    dist_projected_sort = torch.sort(dist_projected, dim = 0).values
    test_projected_sort = torch.sort(test_projected, dim = 0).values

    #return ((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).mean()
    #print(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).shape)
    return torch.sqrt(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1)).mean() / dirNum

def two_sample_wasserstein(sample1, sample2, dirNum):
    Ndim_samp = sample1.shape[1]
    Nsamp = np.min((sample1.shape[0], sample2.shape[0]))
    sampled_directions = torch.randn(dirNum, Ndim_samp)
    sampled_directions = sampled_directions / torch.linalg.norm(sampled_directions, dim = 1).view(sampled_directions.shape[0],1)
    
    samp1_index = range(sample1.shape[0])
    samp2_index = range(sample2.shape[0])
    
    sampled_1 = sample1[random.sample(samp1_index, Nsamp),:]
    sampled_2 = sample2[random.sample(samp2_index, Nsamp),:]
    
    samp1_projected = torch.matmul(sampled_1, sampled_directions.t())
    samp2_projected = torch.matmul(sampled_2, sampled_directions.t())
    
    samp1_projected_sort = torch.sort(samp1_projected, dim = 0).values
    samp2_projected_sort = torch.sort(samp2_projected, dim = 0).values
    
    return torch.sqrt(((samp1_projected_sort - samp2_projected_sort)**2).sum(dim = 1)).mean() / dirNum


class scoreFunction(nn.Module):
    def __init__(self, inputDim, referenceDim, Nnodes):
        super().__init__()

        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.referenceDim = referenceDim
        self.fc1 = nn.Linear(referenceDim + inputDim + 1, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)

    def forward(self, source, target, time):
        z = torch.cat((source, target, time), dim = 1)

        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z)
                
        return z
    
    def ode_forward(self, source_use, initial_condition, dt):

        output = torch.zeros(int(1/dt), initial_condition.shape[0], initial_condition.shape[1])
        output[0,:,:] = initial_condition
        t_use = torch.tensor(0).type(torch.float32)

        t_use = t_use.repeat(repeats = (initial_condition.shape[0],1))

        for i in range(int(1/dt) - 1):
            dx = self.forward(source_use, output[i,:,:].view(initial_condition.shape[0], initial_condition.shape[1]), t_use)
            output[i+1,:,:] = output[i,:,:] + dt * dx

            t_use += dt

        return output


def ode_forward(model, source_use, initial_condition, dt):

        output = torch.zeros(int(1/dt), initial_condition.shape[0], initial_condition.shape[1])
        output[0,:,:] = initial_condition
        t_use = torch.tensor(0).type(torch.float32)

        t_use = t_use.repeat(repeats = (initial_condition.shape[0],1))

        for i in range(int(1/dt) - 1):
            dx = model.forward(source_use, output[i,:,:].view(initial_condition.shape[0], initial_condition.shape[1]), t_use)
            output[i+1,:,:] = output[i,:,:] + dt * dx

            t_use += dt

        return output


def train_model(data, Nnodes = 500, latent_dim = 10, fix = True, batchSize = 150, Ndir = 50, Niter = 30000, balance = False, neg_ratio = 10, fixval = 0.05, matchZ = False, positive = False, wday = 0.0001):
    fix = fix
    
    VAE = simpleVAE(inputDim=data.shape[1], Nnodes = Nnodes, latent_dim =latent_dim, fixStd=fix, fixval=fixval)

    if positive == True:
        VAE = simpleVAE_positive(inputDim=data.shape[1], Nnodes = Nnodes, latent_dim =latent_dim, fixStd=fix, fixval=fixval)
        print('use positive VAE')


    optimizer_VAE = torch.optim.Adam(VAE.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=wday) 

    record_pcf = np.array([])
    record_mov = np.array([])
    record_recon = np.array([])
    
    batchSize = batchSize
    Ndir = Ndir
    mvn = torch.distributions.MultivariateNormal(torch.zeros(VAE.latentDim), torch.eye(VAE.latentDim))

    for i in range(Niter):
        
        optimizer_VAE.zero_grad()


        random_sample = random.sample(range(data.shape[0]),batchSize)
        x = data[random_sample,:]  

        x_hat = VAE(x)
        loss_pcf = torch.tensor([0.0,0.0])
        loss_movement = torch.tensor([0.0,0.0])
        # recon_loss = torch.log(((x-x_hat)**2).sum(dim = 1)).mean()
        recon_loss = ((x-x_hat)**2).sum(dim = 1).mean()
        x_mean = x.mean(dim = 0)
        if matchZ == True:
            x_mean = x.mean(dim = 0)
            x_std = x.std(dim = 0)
            x_std[torch.isnan(x_std)] = 1
            x_std[torch.isinf(x_std)] = 1
            x_std[x_std == 0] = 1


            x = (x - x_mean.view(1,x.shape[1])) / x_std.view(1,x.shape[1])
            x[x > 3] = 3 # clip or not?
            x_hat = (x_hat - x_mean.view(1,x.shape[1])) / x_std.view(1,x.shape[1])



            
        if fix == False:
            loss = ((x-x_hat)**2).sum(dim = 1).mean() +  1 * VAE.KL_loss.mean()
        if fix == True:
            W_dist = sample_wasserstein(mvn, VAE.lowdim, Ndir)
              #+ 300 * loss_movement.mean()#ori
            if balance == True:
                idx = (x>0).flatten()
                n_pos = idx.sum()
                neg_pos = torch.randperm(len(idx))[:max(len(idx), int(n_pos * neg_ratio))]
                
                idx[neg_pos] = True
                idx = idx.view(x.shape[0], x.shape[1])
                loss = ((x-x_hat)**2)[idx].sum() + 5000 * W_dist

            if balance == False:
                loss = ((x-x_hat)**2).sum(dim = 1).mean() + 5000 * W_dist

            
            record_recon = np.append(record_recon, ((x-x_hat)**2).sum(dim = 1).mean().detach().numpy())
        loss.backward()
        optimizer_VAE.step()

        if np.mod(i, 1000) == 0:
            print('Iter: {}, loss_re: {:.4f}, loss_pcf: {:.4f}'
                  .format(i, loss, loss_pcf.abs().mean().detach().numpy()))    
            
            print('reconstruction loss = ' + str(((x-x_hat)**2).sum(dim = 1).mean().detach().numpy()))

            # sns.scatterplot(x = x.flatten().detach().numpy(), y = x_hat.flatten().detach().numpy(), s = 5,
            #                hue = np.tile(x_mean.detach().numpy(), x.shape[0]), palette = 'icefire', legend = True, linewidth = 0)
            
            # # sns.scatterplot(x = x[0].flatten().detach().numpy(), y = x_hat[0].flatten().detach().numpy(), s = 50,
            #             #    hue = x_mean.detach().numpy(), palette = 'icefire', legend = True, linewidth = 0)
            # plt.xlabel('observed')
            # plt.ylabel('predicted')
            # plt.axline((0,0), slope = 1, c = 'k')
            # plt.gcf().set_size_inches(4,4)
            # plt.show()
            clear_output(wait=True)
            
    return VAE


def train_model_matchZ(data, Nnodes = 500, latent_dim = 10, fix = True, batchSize = 150, Ndir = 50, Niter = 30000, balance = False, neg_ratio = 10, fixval = 0.05):
    fix = fix
    
    VAE = simpleVAE(inputDim=data.shape[1], Nnodes = Nnodes, latent_dim =latent_dim, fixStd=fix, fixval=fixval)
    optimizer_VAE = torch.optim.Adam(VAE.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=0.0001) 

    record_pcf = np.array([])
    record_mov = np.array([])
    record_recon = np.array([])
    
    batchSize = batchSize
    Ndir = Ndir
    mvn = torch.distributions.MultivariateNormal(torch.zeros(VAE.latentDim), torch.eye(VAE.latentDim))

    x_mean = data.mean(dim = 0)
    x_std = data.std(dim = 0)
    x_std[torch.isnan(x_std)] = 1
    x_std[torch.isinf(x_std)] = 1
    x_std[x_std == 0] = 1


    for i in range(Niter):
        
        optimizer_VAE.zero_grad()


        random_sample = random.sample(range(data.shape[0]),batchSize)
        x = data[random_sample,:]  

        x_hat = VAE(x)
        loss_pcf = torch.tensor([0.0,0.0])
        loss_movement = torch.tensor([0.0,0.0])
        # recon_loss = torch.log(((x-x_hat)**2).sum(dim = 1)).mean()
        recon_loss = ((x-x_hat)**2).sum(dim = 1).mean()
        x_mean = x.mean(dim = 0)
        
        x_z = (x - x_mean.view(1,x.shape[1])) / x_std.view(1,x.shape[1])
        x_hat_z = (x_hat - x_mean.view(1,x.shape[1])) / x_std.view(1,x.shape[1])

            
        if fix == False:
            loss = ((x-x_hat)**2).sum(dim = 1).mean() +  1 * VAE.KL_loss.mean()
        if fix == True:
            W_dist = sample_wasserstein(mvn, VAE.lowdim, Ndir)
              #+ 300 * loss_movement.mean()#ori
            if balance == True:
                idx = (x>0).flatten()
                n_pos = idx.sum()
                neg_pos = torch.randperm(len(idx))[:max(len(idx), int(n_pos * neg_ratio))]
                
                idx[neg_pos] = True
                idx = idx.view(x.shape[0], x.shape[1])
                loss = ((x-x_hat)**2).flatten()[idx].sum() + 5000 * W_dist + ((x_z-x_hat_z)**2)[idx].sum()

            if balance == False:
                loss = 0 * ((x-x_hat)**2).sum(dim = 1).mean() + 5000 * W_dist + 20 * ((x_z-x_hat_z)**2).sum(dim = 1).mean()

            
            record_recon = np.append(record_recon, ((x-x_hat)**2).sum(dim = 1).mean().detach().numpy())
        loss.backward()
        optimizer_VAE.step()

        if np.mod(i, 1000) == 0:
            print('Iter: {}, loss_re: {:.4f}, loss_pcf: {:.4f}'
                  .format(i, loss, loss_pcf.abs().mean().detach().numpy()))    
            
            print('reconstruction loss = ' + str(((x-x_hat)**2).sum(dim = 1).mean().detach().numpy()))
            print('zscored loss = ' + str(((x_z-x_hat_z)**2).sum(dim = 1).mean().detach().numpy()))

            # sns.scatterplot(x = x.flatten().detach().numpy(), y = x_hat.flatten().detach().numpy(), s = 5,
            #                hue = np.tile(x_mean.detach().numpy(), x.shape[0]), palette = 'icefire', legend = True, linewidth = 0)
            
            # # sns.scatterplot(x = x[0].flatten().detach().numpy(), y = x_hat[0].flatten().detach().numpy(), s = 50,
            #             #    hue = x_mean.detach().numpy(), palette = 'icefire', legend = True, linewidth = 0)
            # plt.xlabel('observed')
            # plt.ylabel('predicted')
            # plt.axline((0,0), slope = 1, c = 'k')
            # plt.gcf().set_size_inches(4,4)
            # plt.show()
            clear_output(wait=True)
            
    return VAE

    
    
    
def get_gene_embedding_ID(model, all_data_tensor, Ntrial = 50000, Nclus = 2):
    mvn = torch.distributions.MultivariateNormal(torch.zeros(model.latentDim), torch.eye(model.latentDim))

    all_hat = model(all_data_tensor)
    all_grad = np.zeros([Ntrial, all_data_tensor.shape[1]])
    loc = np.zeros([Ntrial, model.latentDim])
    DIR = np.zeros([Ntrial, model.latentDim])
    mag = np.array([])
    dir_norm = np.array([])
    cell_idx = np.array([])
    
    for i in range(Ntrial):
        ssample = random.sample(range(all_data_tensor.shape[0]),1)

        n = mvn.sample([1]) * 1 #* 0.0001 *
        dir1 = np.random.randint(low = 0, high = model.latentDim, size = 1)[0]
        n = n/torch.sqrt((n**2).sum()) * 0.1 #ori is 0.1, 
        # n[0,dir1] = 0.1
        z = model.lowdim[ssample, :].detach()
        out = model.back2high(z)
        out_hat = model.back2high(z + n)

        exp_change = out_hat - out
        # exp_change[exp_change.abs() < exp_change.abs().median()] = 0
        # scale_factor = 
        # exp_change = exp_change / exp_change.std()
        all_grad[i,:] = exp_change.detach().numpy() 
        loc[i,:] = z
        DIR[i,:] = n
        mag = np.append(mag, (exp_change**2).detach().numpy().sum())
        dir_norm = np.append(dir_norm, (n.detach().numpy()**2).sum())
        cell_idx = np.append(cell_idx, ssample)
        
    reducer_gene = UM.UMAP(n_neighbors= 5, 
                   metric = 'euclidean', 
                   repulsion_strength=1, 
                   n_epochs= 1500, 
                   negative_sample_rate=5, 
                   min_dist = .1, #ori was 0.5? but 0.1 works better for real scrna-seq data
                   verbose = False,
                   random_state=1,
                   n_components = 2
                  )

    pca_perturb = PCA(n_components=20, random_state = 1)  # smaller number of pcs give clearer cluster?, ori is 50
    raw_data = all_grad.copy()
    data =  np.abs(all_grad.T)
    data = zscore(data)
    store = data.copy()
    data = pca_perturb.fit_transform(data)

    embedding_grad = reducer_gene.fit_transform(data) #original did not use abs
    
    kmeans = KMeans(n_clusters=Nclus, random_state=0)
    spectral = SpectralClustering(n_clusters=Nclus,
            assign_labels='discretize',
            random_state=0)
    kmeans.fit(embedding_grad)
    # spectral.fit(embedding_grad)
    
    return mag, raw_data, embedding_grad, cell_idx

def get_gene_embedding_ID_sampling_based(model, all_data_tensor, Ntrial = 50000, Nclus = 2):
    mvn = torch.distributions.MultivariateNormal(torch.zeros(model.latentDim), torch.eye(model.latentDim))

    all_hat = model(all_data_tensor)
    all_grad = np.zeros([Ntrial, all_data_tensor.shape[1]])
    loc = np.zeros([Ntrial, model.latentDim])
    DIR = np.zeros([Ntrial, model.latentDim])
    mag = np.array([])
    dir_norm = np.array([])
    cell_idx = np.array([])
    
    for i in range(Ntrial):
        ssample = random.sample(range(all_data_tensor.shape[0]),1)

        n = mvn.sample([1]) * 1 #* 0.0001 *
        dir1 = np.random.randint(low = 0, high = model.latentDim, size = 1)[0]
        n = n/torch.sqrt((n**2).sum()) * 0.1 #ori is 0.1, 
        # n[0,dir1] = 0.1
        z = 3 * mvn.sample([1]).detach()
        out = model.back2high(z)
        out_hat = model.back2high(z + n)

        exp_change = out_hat - out
        # exp_change[exp_change.abs() < exp_change.abs().median()] = 0
        # scale_factor = 
        # exp_change = exp_change / exp_change.std()
        all_grad[i,:] = exp_change.detach().numpy() 
        loc[i,:] = z
        DIR[i,:] = n
        mag = np.append(mag, (exp_change**2).detach().numpy().sum())
        dir_norm = np.append(dir_norm, (n.detach().numpy()**2).sum())
        cell_idx = np.append(cell_idx, ssample)
        
    reducer_gene = UM.UMAP(n_neighbors= 5, 
                   metric = 'euclidean', 
                   repulsion_strength=1, 
                   n_epochs= 1500, 
                   negative_sample_rate=5, 
                   min_dist = .1, #ori was 0.5? but 0.1 works better for real scrna-seq data
                   verbose = False,
                   random_state=1,
                   n_components = 2
                  )

    pca_perturb = PCA(n_components=20, random_state = 1)  # smaller number of pcs give clearer cluster?, ori is 50
    raw_data = all_grad.copy()
    data =  np.abs(all_grad.T)
    data = zscore(data)
    store = data.copy()
    data = pca_perturb.fit_transform(data)

    embedding_grad = reducer_gene.fit_transform(data) #original did not use abs
    
    kmeans = KMeans(n_clusters=Nclus, random_state=0)
    spectral = SpectralClustering(n_clusters=Nclus,
            assign_labels='discretize',
            random_state=0)
    kmeans.fit(embedding_grad)
    # spectral.fit(embedding_grad)
    
    return mag, raw_data, embedding_grad, cell_idx

class latent_module_mean(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = self.fc4(z)
        
        return z

class latent_module_std(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        
        return z
    
class back2high(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self,z):
        ori_data = torch.relu(self.fc1(z))
        ori_data = torch.relu(self.fc2(ori_data))
        ori_data = torch.relu(self.fc3(ori_data))
        ori_data = self.fc4(ori_data)
        
        return ori_data
    
class latent_module_mean_dropout(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        m = nn.Dropout(p=0.2)

        z = m(torch.relu(self.fc1(z)))
        z = m(torch.relu(self.fc2(z)))
        z = self.fc4(z)
        
        return z
    
class back2high_positive(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self,z):
        m1 = nn.LeakyReLU(0.1) #nn.Sigmoid()
        m2 = nn.Sigmoid()
        ori_data = m1(self.fc1(z))
        ori_data = m1(self.fc2(ori_data))
        ori_data = m1(self.fc3(ori_data))
        ori_data = m2(self.fc4(ori_data))
        # ori_data = torch.exp(self.fc4(ori_data))
        
        return ori_data
    
class simpleVAE_positive(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim, fixStd = True, fixval = 0.5):
        super().__init__()
        
        self.latentMean = latent_module_mean(inputDim, Nnodes, latent_dim)
        self.latentStd = latent_module_std(inputDim, Nnodes, latent_dim)
        self.back2high = back2high_positive(inputDim, Nnodes, latent_dim)
        
        self.latentDim = latent_dim
        self.fixStd = fixStd
        self.fixVal = fixval
        
        
        
    def forward(self,z):
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim), torch.eye(self.latentDim))
        
        latent_mean = self.latentMean(z)
        latentStd = torch.clamp(self.latentStd(z), min=1e-3)
        
        if self.fixStd == False:
            latent_state = latent_mean + mvn.sample([1]) * latentStd
        
        if self.fixStd == True:
            latent_state = latent_mean + mvn.sample([1]) * self.fixVal

        ori_data = self.back2high(latent_state)
        
        
        self.KL_loss = 0.5 *  ((latent_mean**2).sum(dim = 1)  - torch.clamp(torch.log(latentStd**2), min = -100).sum(dim = 1) - (self.latentDim - 1) + (latentStd**2).sum(dim = 1) )
        self.lowdim = latent_mean
        self.lowstd = latentStd
        return ori_data
    
class simpleVAE(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim, fixStd = True, fixval = 0.5):
        super().__init__()
        
        self.latentMean = latent_module_mean(inputDim, Nnodes, latent_dim)
        self.latentStd = latent_module_std(inputDim, Nnodes, latent_dim)
        self.back2high = back2high(inputDim, Nnodes, latent_dim)
        
        self.latentDim = latent_dim
        self.fixStd = fixStd
        self.fixVal = fixval
        
        
        
    def forward(self,z):
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim), torch.eye(self.latentDim))
        
        latent_mean = self.latentMean(z)
        latentStd = torch.clamp(self.latentStd(z), min=1e-3)
        
        if self.fixStd == False:
            latent_state = latent_mean + mvn.sample([1]) * latentStd
        
        if self.fixStd == True:
            latent_state = latent_mean + mvn.sample([1]) * self.fixVal

        ori_data = self.back2high(latent_state)
        
        
        self.KL_loss = 0.5 *  ((latent_mean**2).sum(dim = 1)  - torch.clamp(torch.log(latentStd**2), min = -100).sum(dim = 1) - (self.latentDim - 1) + (latentStd**2).sum(dim = 1) )
        self.lowdim = latent_mean
        self.lowstd = latentStd
        return ori_data
    
    # def train_VAE(self, train_data, pcf_loss, batchSize = 150, Ndir = 50, Nitr = 20000, pcf_weight = 150):
    #     fix = self.fixStd
    #     pcf = pcf_loss
    #     record_pcf = np.array([])
    #     record_pcf_max = np.array([])
    #     optimizer_VAE = torch.optim.Adam(self.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=0.0001) 

    #     mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim), torch.eye(self.latentDim))

    #     for i in range(Nitr):
    #         optimizer_VAE.zero_grad()


    #         random_sample = random.sample(range(train_data.shape[0]),batchSize)
    #         x = train_data[random_sample,:]
    #         x_hat = self.forward(x)
    #         recon_loss = torch.log(((x-x_hat)**2).sum(dim = 1)).mean()
            
    #         loss_pcf = torch.tensor([0.0,0.0])


    #         if fix == False:
    #             loss = recon_loss +  self.latentDim * self.KL_loss.mean()
    #         if fix == True:
    #             if pcf == True:
    #                 if np.mod(i, 5) == 0:
    #                     loss_pcf = ortho_loss_numerical_v2(self,self.lowdim , eps = 0.1)
    #                     record_pcf = np.append(record_pcf, loss_pcf.abs().mean().detach().numpy())
    #                     record_pcf_max = np.append(record_pcf_max, loss_pcf.abs().max().detach().numpy())

    #             else:
    #                 if np.mod(i, 5) == 0:
    #                     loss_pcf_record = ortho_loss_numerical_v2(self,self.lowdim , eps = 0.1)
    #                     record_pcf = np.append(record_pcf, loss_pcf_record.abs().mean().detach().numpy())
    #                     record_pcf_max = np.append(record_pcf_max, loss_pcf_record.abs().max().detach().numpy())
                        
    #             W_dist = sample_wasserstein(mvn, self.lowdim, Ndir)
    #             # loss = recon_loss + 5000 * W_dist + pcf_weight * loss_pcf.abs().mean()
    #             loss = ((x-x_hat)**2).sum(dim = 1).mean() + 5000 * W_dist + pcf_weight * loss_pcf.abs().mean() #ori
                
    #         loss.backward()
    #         optimizer_VAE.step()

    #         if np.mod(i, 1000) == 0:
    #             print('Iter: {}, loss_re: {:.4f}'
    #                   .format(i, recon_loss, ))    
    #             # clear_output(wait=True)
                
                
        # train_hat = self.forward(train_data)
        # # pred_accuracy = np.corrcoef(train_data.flatten().detach().numpy(), train_hat.flatten().detach().numpy())[0,1]
        # pred_accuracy = ((train_data-train_hat)**2).sum(dim = 1).mean()
        # self.accuracy = pred_accuracy
        # self.record_pcf = record_pcf
        # self.record_pcf_max = record_pcf_max

        # return self
    
class MBTA_submodule:
    def __init__(self, model, method_name, modality_name):
        self.model = model
        self.method_name = method_name
        self.modality_name = modality_name

    def predict(self, input_tensor):
        info1 = self.model(input_tensor)
        predicted_values = self.model.back2high(self.model.lowdim).detach().numpy()
        return predicted_values
    
    def correlate(self, input_tensor, Zscore = True):
        predicted_values = self.predict(input_tensor)
        input_tensor = input_tensor.detach().numpy()

        if Zscore == True:
            input_tensor = zscore(input_tensor)
            input_tensor[np.isnan(input_tensor)] = 0
            predicted_values = zscore(predicted_values)

        return np.corrcoef(input_tensor.flatten(), predicted_values.flatten())[0,1]
    
    def correlate_each_sample(self, input_tensor, sample_labels = 'NA', Zscore = True):
        predicted_values = self.predict(input_tensor)
        input_tensor = input_tensor.detach().numpy()

        if Zscore == True:
            input_tensor = zscore(input_tensor)
            input_tensor[np.isnan(input_tensor)] = 0
            predicted_values = zscore(predicted_values)


        half_point = input_tensor.shape[0]

        catData = np.vstack((input_tensor, 
                             predicted_values))

        corr = np.corrcoef(catData)
        corr = corr[:half_point,half_point:]

        df = pd.DataFrame(dict(corr = np.diag(corr),
                               method = self.method_name,
                               modality = self.modality_name,
                               cell_type = sample_labels)
                       )

        return df
    
    def CramerVonMises(self, input_tensor):
        predicted_values = self.predict(input_tensor)
        input_tensor = input_tensor.detach().numpy()

        out = []
        for i in range(predicted_values.shape[1]):
            cvm = scipy.stats.cramervonmises_2samp(predicted_values[:,i], input_tensor[:,i])
            out.append(cvm.statistic)
        
        return np.array(out)

            


    
    ''' 
    all plotting related things below
    '''
    def plot_sample_correlation(self, input_tensor, sample_labels = 'NA', Zscore = True):
        df = self.correlate_each_sample(input_tensor, sample_labels, Zscore)
        sns.kdeplot(df, x = 'corr', hue = 'cell_type', bw_adjust=0.5, fill = True, common_norm=False)


    def plot(self, input_tensor, Zscore = True):
        predicted_values = self.predict(input_tensor)
        input_tensor = input_tensor.detach().numpy()

        if Zscore == True:
            input_tensor = zscore(input_tensor)
            input_tensor[np.isnan(input_tensor)] = 0
            predicted_values = zscore(predicted_values)

        
        plt.scatter(input_tensor, predicted_values, s = 1, alpha = 0.1)



'''
what other modules to add?
1. one that output embedding coordinate of real and generated samples 
2. 
'''



'''
add another module that combines both models,
and then learns the translation between the two models 
'''
class general_MBTA:
    def __init__(self):
        print('class created')
    def add_model(self, model, model_name):
        setattr(self, model_name, model)
    
    def paired_training(self, source, source_model, target, target_model, 
    nnode = 800, batchSize = 100, Nitr = 100000):
        model_name = source_model + '_' + target_model # FROM -> TWO
        info1 = getattr(self, source_model).model(source)
        info2 = getattr(self, target_model).model(target)

        source = getattr(self, source_model).model.lowdim.clone().detach()
        target = getattr(self, target_model).model.lowdim.clone().detach()

        print('done generating source and target')
        print(source.shape)


        model_fm = scoreFunction(target.shape[1], source.shape[1], nnode)

        optimizer = torch.optim.Adam(model_fm.parameters(), lr = 0.0001, betas = (0.8, 0.9), weight_decay=0.000001) 

        
        torch.set_num_threads(4)  # Replace 4 with your desired number of threads

        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))
        record_loss = np.array([])

        for i in range(Nitr):
            optimizer.zero_grad()

            idx = random.sample(range(target.shape[0]), batchSize)
            sampled_time = torch.rand(batchSize).view(batchSize, 1)

            sampled_source = source[idx,:]
            sampled_target = target[idx,:]

            sampled_noise = mvn.sample([batchSize])
            target_X = sampled_target * sampled_time +  (1-sampled_time) * sampled_noise
            target_vec = sampled_target - sampled_noise

            out = model_fm(sampled_source, target_X, sampled_time)

            loss = ((target_vec - out)**2).sum(dim = 1).mean()

            loss.backward()
            optimizer.step()


            record_loss = np.append(record_loss, loss.detach().numpy())

            if np.mod(i, 1000) == 0:
                print(i)

        setattr(self, model_name, model_fm)

    def coarse_training(self, source, source_model, target, target_model,
                        source_label, target_label, nnode = 800, batchSize = 100, Nitr = 100000):
        
        source_label = np.array(source_label) 
        target_label = np.array(target_label) 

        model_name = source_model + '_' + target_model # FROM -> TWO
        info1 = getattr(self, source_model).model(source)
        info2 = getattr(self, target_model).model(target)

        source = getattr(self, source_model).model.lowdim.clone().detach()
        target = getattr(self, target_model).model.lowdim.clone().detach()

        print('done generating source and target')
        print(source.shape)

        model_fm = scoreFunction(target.shape[1], source.shape[1], nnode)

        optimizer = torch.optim.Adam(model_fm.parameters(), lr = 0.0001, betas = (0.8, 0.9), weight_decay=0.000001) 

        
        torch.set_num_threads(4)  # Replace 4 with your desired number of threads

        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))
        record_loss = np.array([])

        for i in range(Nitr):
            optimizer.zero_grad()

            idx1_ini = np.array(random.choices(range(source.shape[0]), k = batchSize))
            idx2_ini = np.array(random.choices(range(target.shape[0]), k = batchSize))

            idx1 = idx1_ini[source_label[idx1_ini] == target_label[idx2_ini]]
            idx2 = idx2_ini[source_label[idx1_ini] == target_label[idx2_ini]]

            use_samp = idx1.shape[0]
            


            sampled_time = torch.rand(use_samp).view(use_samp, 1)

            sampled_source = source[idx1,:]
            sampled_target = target[idx2,:]

            sampled_noise = mvn.sample([use_samp])
            target_X = sampled_target * sampled_time +  (1-sampled_time) * sampled_noise
            target_vec = sampled_target - sampled_noise

            out = model_fm(sampled_source, target_X, sampled_time)

            loss = ((target_vec - out)**2).sum(dim = 1).mean()

            loss.backward()
            optimizer.step()


            record_loss = np.append(record_loss, loss.detach().numpy())

            if np.mod(i, 1000) == 0:
                # print(use_samp)
                print(i)
        setattr(self, model_name, model_fm)

    def generate(self, source, source_model, target, target_model, Nsamp = 5000, dt = 0.01, get_low =False):
        info1 = getattr(self, source_model).model(source)
        info2 = getattr(self, target_model).model(target)

        source = getattr(self, source_model).model.lowdim.clone().detach()
        target = getattr(self, target_model).model.lowdim.clone().detach()

        print('done generating source and target')
        print(target.shape)

        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))
        if source.shape[0] == target.shape[0]:
            use_source = source.clone()
            Nsamp = target.shape[0]
        elif source.shape[0] == target.shape[1]:
            use_source = source.view(1,source.shape[0])
            use_source = use_source.repeat(repeats = (Nsamp, 1))

        model_name = source_model + '_' + target_model # FROM -> TWO
        model_use = getattr(self, model_name)

        inputX = mvn.sample([Nsamp])
        ode_sol = ode_forward(model_use, use_source, inputX, dt)
        reconstructed_modality = getattr(self, target_model).model.back2high(ode_sol[-1,:,:]).detach().numpy()
        
        if get_low == True:
             return reconstructed_modality, ode_sol[-1,:,:]

        if get_low == False:
            return reconstructed_modality 

class paired_MBTA:
    def __init__(self, model1, model2):
        self.m1 = model1
        self.m2 = model2
        self.method_name = model1.method_name
        self.modality1 = model1.modality_name
        self.modality2 = model2.modality_name
    def coarse_pairing(self, input_tensor1, input_tensor2, label1, label2, Dir = 1,
                       nnode = 800, batchSize = 2000, Nitr = 1000000):
        info1 = self.m1.model(input_tensor1)
        info2 = self.m2.model(input_tensor2)

        if Dir == 1:
            source = torch.tensor(self.m1.model.lowdim.clone()).type(torch.float32)
            target = torch.tensor(self.m2.model.lowdim.clone()).type(torch.float32)

            source_label = label1
            target_label = label2

        if Dir == 2:
            target = torch.tensor(self.m1.model.lowdim.clone()).type(torch.float32)
            source = torch.tensor(self.m2.model.lowdim.clone()).type(torch.float32)

            source_label = label2
            target_label = label1


        model_fm = scoreFunction(target.shape[1], source.shape[1], nnode)

        optimizer = torch.optim.Adam(model_fm.parameters(), lr = 0.0001, betas = (0.8, 0.9), weight_decay=0.000001) 

        
        torch.set_num_threads(4)  # Replace 4 with your desired number of threads

        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))
        record_loss = np.array([])

        for i in range(Nitr):
            optimizer.zero_grad()

            idx1_ini = np.array(random.choices(range(source.shape[0]), k = batchSize))
            idx2_ini = np.array(random.choices(range(target.shape[0]), k = batchSize))

            idx1 = idx1_ini[source_label[idx1_ini] == target_label[idx2_ini]]
            idx2 = idx2_ini[source_label[idx1_ini] == target_label[idx2_ini]]

            use_samp = idx1.shape[0]
            


            sampled_time = torch.rand(use_samp).view(use_samp, 1)

            sampled_source = source[idx1,:]
            sampled_target = target[idx2,:]

            sampled_noise = mvn.sample([use_samp])
            target_X = sampled_target * sampled_time +  (1-sampled_time) * sampled_noise
            target_vec = sampled_target - sampled_noise

            out = model_fm(sampled_source, target_X, sampled_time)

            loss = ((target_vec - out)**2).sum(dim = 1).mean()

            loss.backward()
            optimizer.step()


            record_loss = np.append(record_loss, loss.detach().numpy())

            if np.mod(i, 1000) == 0:
                # print(use_samp)
                print(i)

        if Dir == 1:
            self.MBTA_12 = model_fm
            self.loss_12 = record_loss
        if Dir == 2:
            self.MBTA_21 = model_fm
            self.loss_21 = record_loss

            
     
    def paired_training(self, input_tensor1, input_tensor2, Dir = 1,
                            nnode = 800, batchSize = 100, Nitr = 100000):
        info1 = self.m1.model(input_tensor1)
        info2 = self.m2.model(input_tensor2)

        if Dir == 1:
            source = torch.tensor(self.m1.model.lowdim.clone()).type(torch.float32)
            target = torch.tensor(self.m2.model.lowdim.clone()).type(torch.float32)
        if Dir == 2:
            target = torch.tensor(self.m1.model.lowdim.clone()).type(torch.float32)
            source = torch.tensor(self.m2.model.lowdim.clone()).type(torch.float32)


        model_fm = scoreFunction(target.shape[1], source.shape[1], nnode)

        optimizer = torch.optim.Adam(model_fm.parameters(), lr = 0.0001, betas = (0.8, 0.9), weight_decay=0.000001) 

        
        torch.set_num_threads(4)  # Replace 4 with your desired number of threads

        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))
        record_loss = np.array([])

        for i in range(Nitr):
            optimizer.zero_grad()

            idx = random.sample(range(target.shape[0]), batchSize)
            sampled_time = torch.rand(batchSize).view(batchSize, 1)

            sampled_source = source[idx,:]
            sampled_target = target[idx,:]

            sampled_noise = mvn.sample([batchSize])
            target_X = sampled_target * sampled_time +  (1-sampled_time) * sampled_noise
            target_vec = sampled_target - sampled_noise

            out = model_fm(sampled_source, target_X, sampled_time)

            loss = ((target_vec - out)**2).sum(dim = 1).mean()

            loss.backward()
            optimizer.step()


            record_loss = np.append(record_loss, loss.detach().numpy())

            if np.mod(i, 1000) == 0:
                print(i)

        if Dir == 1:
            self.MBTA_12 = model_fm
            self.loss_12 = record_loss
        if Dir == 2:
            self.MBTA_21 = model_fm
            self.loss_21 = record_loss

    def generate(self, source, target, Nsamp = 5000, Dir = 1, dt = 0.01):
        
        '''
        Need to make sure that souce, target matches Dir
        two cases
        1. given one single source, generate a large amount of samples
        2. given one source for each generated sample
        '''
        mvn = torch.distributions.MultivariateNormal(torch.zeros(target.shape[1]), torch.eye(target.shape[1]))

        if Dir == 1:
            model_use = self.MBTA_12
        if Dir == 2:
            model_use = self.MBTA_21

        # if source.shape[0] == target.shape[1]:
        # if source.shape[0] == 1:
        #     use_source = source.view(1,source.shape[0])
        #     use_source = use_source.repeat(repeats = (Nsamp, 1))
        if source.shape[0] == target.shape[0]:
            use_source = source.clone()
            Nsamp = target.shape[0]
        elif source.shape[0] == target.shape[1]:
            use_source = source.view(1,source.shape[0])
            use_source = use_source.repeat(repeats = (Nsamp, 1))

        inputX = mvn.sample([Nsamp])
        ode_sol = ode_forward(model_use, use_source, inputX, dt)

        if Dir == 1:
            reconstructed_modality = self.m2.model.back2high(ode_sol[-1,:,:]).detach().numpy()
        if Dir == 2:
            reconstructed_modality = self.m1.model.back2high(ode_sol[-1,:,:]).detach().numpy()
        
        return reconstructed_modality





        
            



        








    
