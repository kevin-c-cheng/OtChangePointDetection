# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:01:18 2019

@author: Kevin Cheng
"""
import scipy.io as scio
import ot
import scipy.stats as scstats
import sklearn.manifold as mfold
import matplotlib.pyplot as plt
import time
import numpy as np

import sys
sys.path.insert(0,"../Code")
from WassersteinChangePointDetectionLib import ComputeOtDistance
from OtSingleDimStatLib import *
import numpy.random as rd

def GaussianKernelFunc(x,y, sigma=1):
    if (sigma is None):
        sigma = 1
    nSamp = np.shape(x)[0]
    if nSamp == 1:
        return np.exp(-(x-y)**2/(2*sigma))

    if (np.ndim(x)==1):
        x=np.expand_dims(x,0)
    if (np.ndim(y)==1):
        y=np.expand_dims(y,0)
    M = ot.dist(x,y,metric = 'sqeuclidean')
    return np.exp(-M/(2*sigma))

def MmdGaussianKernelFunc(x,y=None,sigma=1, diag=0):
    if (sigma is None):
        sigma = 1
    if (y is None):
        y=x
    dist = GaussianKernelFunc(x,y,sigma=sigma)
    if (diag == 0 ):
        return np.sum(dist)-np.sum(np.diag(dist)) # if we're looking just at x-x, remove diagonal
    else:
        return np.sum(dist) # if we're looking just at x-x, remove diagonal

def ComputeMmdDistance(cloud_i,cloud_j, sigma = None):
    m = cloud_i.shape[0]
    n = cloud_j.shape[0]

    if (sigma is None):
        # Comput the median trick
        fullDat = np.append(cloud_i,cloud_j, axis=0)
        distance = ot.utils.dist(fullDat, fullDat, metric = 'sqeuclidean')
        sig = np.median(distance.flatten())
    
    if (m==n):
        mmd = 1/(m*(m-1))*MmdGaussianKernelFunc(cloud_i, sigma=sigma)+ 1/(n*(n-1))*MmdGaussianKernelFunc(cloud_j, sigma=sigma)-2/(m*(m-1))*MmdGaussianKernelFunc(cloud_i,cloud_j, sigma=sigma)
    else:
        mmd = 1/(m*(m-1))*MmdGaussianKernelFunc(cloud_i, sigma=sigma)+ 1/(n*(n-1))*MmdGaussianKernelFunc(cloud_j, sigma=sigma)-2/(m*n)*MmdGaussianKernelFunc(cloud_i,cloud_j, sigma=sigma, diag=1)
    return mmd # Return square value. 

# Simulation Data
nRepeat=1
nSamp = 500
nWindow = 200 # this is used in simulatedKC
nPhases = 4

window = 100
stride = 1
seed = 0000 #90000 #80000 #70000 #60000 #50000 #40000 #30000 #20000 #1000 #0

# Generate data
count = 0
rd.seed(seed)
dat = np.zeros((nWindow*nPhases*nRepeat, nSamp))
datBaseline = np.zeros((nWindow*nPhases*nRepeat, nSamp*10))
for pp in range(nPhases*nRepeat):
    p = np.mod(pp, nPhases)
    if (p==0):
        dat[count:count+nWindow,:] = rd.normal(0,1,size=(nWindow, nSamp))
    elif(p==1):
        dat[count:count+nWindow,:] = rd.laplace(0,1/np.sqrt(2),size=(nWindow, nSamp))
    elif(p==2):
        dat[count:count+nWindow,:] = rd.normal(0,1.2,size=(nWindow, nSamp))
    elif(p==3):
        dat[count:count+nWindow,:] = rd.normal(0.2,1,size=(nWindow, nSamp))
    count = count+nWindow

dat = np.round_(dat,3)

# Now we compute distance metrics, and clustering for data Compute Distance Metrics
distMmd = np.zeros((len(dat),len(dat)))
distWass = np.zeros((len(dat),len(dat)))
distKS = np.zeros((len(dat),len(dat)))
distw2Samp = np.zeros((len(dat),len(dat)))

t1=time.time()
for i in range(len(dat)):
    for j in range(len(dat)):
        dati2 = np.expand_dims(dat[i],1)
        datj2 = np.expand_dims(dat[j],1)
        distMmd[i,j] = np.abs(ComputeMmdDistance(dati2, datj2, sigma = 1))
        distWass[i,j] = ComputeOtDistance(dati2, datj2)
        distKS[i,j] = scstats.ks_2samp(dati2.flatten(), datj2.flatten())[0]
        distw2Samp[i,j] = TwoSampleWTest(dati2, datj2)
    print(['done with ' + str(i) + ' of ' + str(nWindow*nPhases)])
t2=time.time()

# Compute low dimensional embedding
distMmd = np.maximum(distMmd,0)
mmdEmbed = mfold.TSNE(n_components=2, perplexity=40, metric='precomputed').fit_transform(distMmd)
wassEmbed = mfold.TSNE(n_components=2, perplexity=40, metric='precomputed').fit_transform(distWass)
ksEmbed = mfold.TSNE(n_components=2, perplexity=40, metric='precomputed').fit_transform(distKS)
w2StatEmbed = mfold.TSNE(n_components=2, perplexity=40, metric='precomputed').fit_transform(distw2Samp)

mmdSE = mfold.SpectralEmbedding(n_components=2, n_neighbors=40, affinity='nearest_neighbors').fit_transform(distMmd)
wassSE = mfold.SpectralEmbedding(n_components=2, n_neighbors=40, affinity='nearest_neighbors').fit_transform(distWass)
ksSE = mfold.SpectralEmbedding(n_components=2, n_neighbors=40, affinity='nearest_neighbors').fit_transform(distKS)
w2StatSE = mfold.SpectralEmbedding(n_components=2, n_neighbors=40, affinity='nearest_neighbors').fit_transform(distw2Samp)

scio.savemat('SimulatedDistances_2Samp.mat', mdict={'distMmd':distMmd, 'distWass':distWass,'distKS':distKS, 'distw2Samp':distw2Samp, \
                                                 'mmdEmbed':mmdEmbed, 'wassEmbed':wassEmbed,'ksEmbed':ksEmbed, 'w2StatEmbed':w2StatEmbed, \
                                                 'mmdSE':mmdSE, 'wassSE':wassSE,'ksSE':ksSE, 'w2StatSE':w2StatSE})
