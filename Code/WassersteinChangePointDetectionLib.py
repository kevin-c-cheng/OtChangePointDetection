# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:21:27 2019

@author: Kevin Cheng
"""
# May 15th update: change to block partitioning. Borrow from hierarchical clustering 
# We take each point, merge adjacent segements with minimum wasserstein distance
# Compute wasserstein barycenter for each segement,
# Then again merge adjaacent segments with minimum wasserstein distance,
# Continue until all barycenters have distanece delta from each other. 

import scipy.io as scio
import scipy.signal as scisig
from scipy.stats import norm
import numpy as np
from sklearn.manifold import TSNE
import sklearn.cluster as skClust
import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, sys, os, ot
import pandas as pd
import random as rd
from OtSingleDimStatLib import *
import scipy.stats as scstats

from scipy.stats import wasserstein_distance

def ComputeSinkhornDistance(cloud_i, cloud_j, metric = 'sqeuclidean', scale = True, lambd = 1e-2):
    nSamp = cloud_i.shape[0]
    a = np.ones(nSamp)/nSamp

    M = ot.utils.dist(cloud_i, cloud_j, metric = metric)
    scale = M.max()
    M = M/scale
    
    G = ot.sinkhorn(a,a,M,lambd)
    if (scale == False):
        return np.sum(np.multiply(G,M))
    else:
        return np.sum(np.multiply(G,M))*scale
    
def ComputeOtDistance(cloud_i,cloud_j, metric = 'sqeuclidean'):
    nSamp_i = cloud_i.shape[0]
    nSamp_j = cloud_j.shape[0]
    a = np.ones(nSamp_i)/nSamp_i
    b = np.ones(nSamp_j)/nSamp_j
    return ComputeOtDistanceWeight(cloud_i, cloud_j, a, b, metric = metric)
#    M = ot.utils.dist(cloud_i,cloud_j, metric = metric)
#    G = ot.emd(a,b,M)
#    return np.sqrt(np.sum(np.multiply(G,M)))

def ComputeOtDistanceWeight(c1, c2, w1, w2, metric = 'sqeuclidean'):
    M = ot.utils.dist(c1,c2, metric = metric)
    G = ot.emd(w1,w2,M)
    return np.sqrt(np.sum(np.multiply(G,M)))

def ComputeSinkhornDistance_variableLambd(cloud_i, cloud_j, metric = 'sqeuclidean'):
    lambdStart = 1e-10
    nSamp = cloud_i.shape[0]
    a = np.ones(nSamp)/nSamp

    M = ot.utils.dist(cloud_i, cloud_j, metric = metric)
    
    converged = 0
    lambd = lambdStart
    while (converged == 0):
        G = ot.sinkhorn(a,a,M,lambd)
        if (np.sum(G)==0):
            lambd = lambd*10
        else:
            converged = 1
    return np.sum(np.multiply(G,M))


def WassersteinBarycenterPartition(data, cp, length = None):
    # Now we define wasserstein barycenter as all the samples generated
    #  within the change points
    if (length is not None):
        cp = np.append(cp, length)
    nBary = len(cp)
    baryOut = []
    cur = 0
    for i in range(nBary):
        baryOut.append(data[cur:cp[i]])
        cur=cp[i]
    return baryOut

def WassersteinBarycenterPartitionWeighted(data, cp, window = None, length= None):
    if (length is not None):
        cp = np.append(cp, length)
    nBary = len(cp)
    baryOut = []
    weightOut = []
    cur = 0
    for i in range(nBary):
        baryOut.append(data[cur:cp[i]])
        if (window is None):
            win = np.ones(cp[i]-cur)
            win = win/np.sum(win)
            weightOut.append(win)
        else:
            if (cp[i]-cur >2*window): #if we're bigger than 2x window
                ham = sp.signal.windows.hamming(2*window)
                win = np.ones(len(data[cur:cp[i]]))
                wMin = min(window, len(win))
                win[:window] = win[:wMin]*ham[:wMin]
                win[-window:] = win[-wMin:]*ham[-wMin:]
            else: #if we're smaller than 2x window, just create a hamming window
                win = sp.signal.window.hamming(cp[i]-cur) 
            win = win/np.sum(win)
            weightOut.append(win)
        cur=cp[i]
    return (baryOut, weightOut)

def WassersteinBarycenterPartitionWeightedNAtom(data, cp, nAtom, window = None, length= None):
    if (length is not None):
        cp = np.append(cp, length)
    nBary = len(cp)
    baryOut = []
    weightOut = []
    cur = 0
    for i in range(nBary):
        if (window is None):
            win = np.ones(cp[i]-cur)
            win = win/np.sum(win)
            winOut = np.ones(nAtom)/nAtom
        else:
            ham = sp.signal.windows.hamming(2*window)
            win = np.ones(len(data[cur:cp[i]]))
            wMin = min(window, len(win))
            win[:window] = win[:wMin]*ham[:wMin]
            win[-window:] = win[-wMin:]*ham[-wMin:]
            win = win/np.sum(win)
            winOut = sp.signal.windows.hamming(nAtom)
            winOut = winOut/np.sum(winOut)
        X_init = np.ones((nAtom,1))*data[cur] + np.random.normal(0,0.1,(nAtom,1))
        weightOut.append(winOut)
        tmp1=[]
#        for d in data[cur:cp[i]]:
#            tmp1.append(np.expand_dims(d,axis=1))
        tmp1.append(data[cur:cp[i]])
#        tmp1.append(data[cur:cp[i]])
        tmp2=[]
        tmp2.append(win)
#        tmp2.append(win)
#        baryOut.append(ot.lp.free_support_barycenter(data[cur:cp[i]], win, X_init, b=winOut))
        baryOut.append(ot.lp.free_support_barycenter(tmp1, tmp2, X_init, b=winOut))
#        baryOut.append(ot.lp.free_support_barycenter(tmp1, np.array([1]), X_init))
        cur=cp[i]
    return (baryOut, weightOut)


def WassersteinAffinity(dat, gamma = 1.0):
    # compute the affinity matrix between samples in dat
    # exp(-gamma*d(xi,xj))
    nDist = len(dat)
    dist = np.zeros((nDist,nDist))
    for i in range(nDist):
        for j in range(nDist):
            if (i==j):
                dist[i,j] = 0
            else:
                dist[i,j] = ComputeOtDistance(dat[i],dat[j])
    return np.exp(-gamma*dist**2) # This is the default for scikitlearn spectral clustering

def WassersteinSpectralClustering_affinity(dat, K):
    dist = WassersteinAffinity(dat)
    clustering = SpectralClustering(n_clusters=K, assign_labels="discretize",random_state=0, affinity='precomputed').fit(dist)
    return clustering.labels_


def WassersteinSpectralClustering_neighbors(dat, K, nNeighbors):
    nDist = len(dat)
    dist = WassersteinAffinity(dat)
    distOut = np.zeros((nDist,nDist))
    for i in range(nDist):
        ss = np.argsort(dist[i])
        distOut[i,ss[-(nNeighbors+1):]] = 1
    distOut = 0.5 * (distOut + distOut.T)
    distOut = distOut - np.eye(nDist)
    
    clustering = SpectralClustering(n_clusters=K, affinity = 'precomputed').fit(distOut)
    return clustering.labels_


def ComputeBlockDistances(blocks, weights = None, log = False):
#    nB = blocks.shape[0]
    nB = len(blocks)
    dOut = np.zeros([nB, nB])
    for i in range(nB):
        for j in range(i,nB):
#            dOut[i,j] = ComputeSinkhornDistance_variableLambd(blocks[i,:,:], blocks[j,:,:])
#            dOut[i,j] = ComputeSinkhornDistance(blocks[i,:,:], blocks[j,:,:],scale=False)
            if (weights is None):
                dOut[i,j] = ComputeOtDistance(blocks[i], blocks[j])
            else:
                dOut[i,j] = ComputeOtDistanceWeight(blocks[i], blocks[j], weights[i], weights[j])
                
            dOut[j,i] = dOut[i,j]
        if (log == True):
            print("done for " + str(i+1) + " out of " + str(nB))
    return dOut

def LogTime(pt):
    ct = time.time()
    print(str(ct-pt))
    pt = ct
    return pt

def w2SampClustering(accelDatO, name, window, stride, nSeg, convFilter, outPath = '', th=0.462, w2Samp=None, useGtCP = False):
    #here we compute wstat, find peaks, partition data, and wasserstein barycenter KMeans on it, and visualize using tsne
    pt = time.time()
    runDebug = False
    saveTmp = True
    log = False

    if (log): sys.stdout.write("w2Samp CPD liklihood...")
    debugFile = outPath + 'tmp_'+name+'_wStat.mat'
    if (runDebug and os.path.isfile(debugFile)):
        if (log): sys.stdout.write('  temporary file... Loading ... ')                
        w2Samp = scio.loadmat(debugFile)["w2Samp"].flatten()
        w2SampC = scio.loadmat(debugFile)["w2SampC"].flatten()
        wStatPks = scio.loadmat(debugFile)["wStatPks"].flatten()
    else:
        if (w2Samp is None):
            (w2Samp, dump) = Compute2SampleWStat(accelDatO, window, stride) 
            
        if (useGtCP == False):
            w2SampC = np.convolve(w2Samp, convFilter, mode = 'same')
            pkIdx = scisig.find_peaks(w2SampC, height=th, prominence=np.max(w2SampC)/1000, width=2)
        else:
            w2SampC = w2Samp
            pkIdx = scisig.find_peaks(w2SampC, height=th, prominence=np.max(w2SampC)/1000)
        wStatPks = pkIdx[0]
        if (saveTmp == True):
            scio.savemat(debugFile, mdict={'w2Samp':w2Samp, 'w2SampC':w2SampC, 'wStatPks':wStatPks})
    if (log): pt = LogTime(pt)

    if (log): sys.stdout.write("wStat Barycenter Computation...")
    debugFile = outPath + 'tmp_'+name+'_barycenter.mat'
    if (runDebug and os.path.isfile(debugFile)):
        if (log): sys.stdout.write('  temporary file... Loading ... ')                
        partBary_p = scio.loadmat(debugFile)["partBary_p"]
        partBary_w = scio.loadmat(debugFile)["partBary_w"]
    else:
#        (partBary_p, partBary_w) = WassersteinBarycenterPartitionWeighted(accelDatO, wStatPks, window=window, length=len(accelDatO))
        (partBary_p, partBary_w) = WassersteinBarycenterPartitionWeighted(accelDatO, wStatPks, length=len(accelDatO))
#        (partBary_p, partBary_w) = WassersteinBarycenterPartitionWeightedNAtom(accelDatO, wStatPks, window, length=len(accelDatO))
        if (saveTmp == True):
            scio.savemat(debugFile, mdict={'partBary_p':partBary_p,'partBary_w':partBary_w})
    if (log): pt = LogTime(pt)
    
    if (log): sys.stdout.write("Compute Partition Barycenter Distances...")
    debugFile = outPath + 'tmp_'+name+'_ComputeBlockDistances.mat'
    if (runDebug and os.path.isfile(debugFile)):
        if (log):
            sys.stdout.write('  temporary file... Loading ... ')                
        barycenterDistance = scio.loadmat(debugFile)["barycenterDistance"]
    else:
        barycenterDistance = ComputeBlockDistances(partBary_p, weights=partBary_w)    
        if (saveTmp == True):
            scio.savemat(debugFile, mdict={'barycenterDistance':barycenterDistance})
    if (log): pt = LogTime(pt)

    if (log): sys.stdout.write("Wasserstein Barycenter Spectral Clustering ...")
    debugFile = outPath + 'tmp_'+name+'_BarycenterSpect.mat'
    if (runDebug and os.path.isfile(debugFile)):
        if (log): sys.stdout.write('  temporary file... Loading ... ')                
        assign = scio.loadmat(debugFile)["assign"].flatten()
    else:
        spCluster = skClust.SpectralClustering(n_clusters=np.minimum(len(partBary_p),nSeg), affinity='precomputed').fit(np.exp(-barycenterDistance))
        assign = spCluster.labels_ +1
        if (saveTmp == True):
            scio.savemat(debugFile, mdict={'assign':assign})
    if (log): pt = LogTime(pt)
    
    return (w2Samp,w2SampC, wStatPks, assign, partBary_p, partBary_w)

