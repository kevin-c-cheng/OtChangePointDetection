# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:51:04 2019

@author: Kevin Cheng
"""

import numpy as np
import scipy.io as scio
from WassersteinChangePointDetectionLib import *
from DataSetParameters import *
from ChangePointMetrics import *
import sys, os
import warnings

if __name__ == '__main__':
    try:
        dataSet = sys.argv[1]
    except:
#        dataSet = "Beedance"
        dataSet = "HASC"
#        dataSet = "HASC2016"
#        dataSet = "UCR"
        
    
    if (dataSet == "Other"):
        fold = sys.argv[2]
        window = int(sys.argv[3])        
        stride = int(sys.argv[4])
        nClust = int(sys.argv[5])
        delay = int(sys.argv[6])
        fMax = float(sys.argv[7])
        precomputed = False
        params = ImportParameters(fold, window, stride, nBack, nClust, delay, pMatch, precomputed, stack, fMax)
        nDat = params.nDat
        beeFiles=params.files
    else:
        params = GetDataParameters(dataSet, os.getcwd()+'\\..\\Data\\')
        nDat = params.nDat
        window = params.window
        stride = params.stride
        nClust = params.nClust
        delay = params.delay
        precomputed = params.precomputed
        beeFiles = params.files

    useGtCP = False
    dispWarn = False
    if (dispWarn is False):
        warnings.filterwarnings("ignore")

    
    w2ConvFilter = scio.loadmat("TwoSampConvFilter.mat")["filter2"].flatten()
    w2ConvFilter = w2ConvFilter[0::int(np.ceil(len(w2ConvFilter)/(2*window)))]-0.166
    w2ConvFilter = w2ConvFilter / np.sum(w2ConvFilter)
    
    dTrain = list(range(nDat))
    dTest=[]
    
    # Now compute for various thresholds
    w2SampStore=[None] *len(dTrain)
    nPartition = np.zeros(len(dTrain))
    
    alphaVecW2 = np.linspace(0,1,num=10)
    alphaVecMmd = np.linspace(0,8,num=10)
    aIdx = 0

    datOAll=[]
    datAll=[]
    labAll=[]
    clustLabAll=[]
    gtCpAll=[]
    w2SampAll = []
    mmdSampAll = []
    for i in range(len(dTrain)):
    #load all data files
        if (precomputed is False and useGtCP==False):
            datOAll.append((scio.loadmat(beeFiles[dTrain[i]])["Y"]).astype(float))
            datAll.append(scio.loadmat(beeFiles[dTrain[i]])["Y"][window:-window])
            labAll.append(scio.loadmat(beeFiles[dTrain[i]])["L"].flatten()[window:-window])
            clustLabAll.append(scio.loadmat(beeFiles[dTrain[i]])["Lc"].flatten()[window:-window])
        else:
            datAll.append((scio.loadmat(beeFiles[dTrain[i]])["Y"][window:-window]).astype(float))#computing the statistic makes us toss out the first (window) values
            labAll.append(scio.loadmat(beeFiles[dTrain[i]])["L"].flatten()[window:-window])
            clustLabAll.append(scio.loadmat(beeFiles[dTrain[i]])["Lc"].flatten()[window:-window])                

            if (useGtCP == False):
                w2SampAll.append(scio.loadmat(beeFiles[dTrain[i]])["wStat2Samp"].flatten()) 
                    
            else:
                #Hardcode change points
                w2SampAll.append(scio.loadmat(beeFiles[dTrain[i]])["L"].flatten()[window:-window]) 
    
    aCount=0
    #Run for a variety of threshold values
    for aCount in range(len(alphaVecW2)):
        gtCpAll=[]
        gtLabAll=[]
        w2CpAll = []
        w2LabAll = []
        w2SampCAll = []
        for i in range(len(dTrain)):
            dat = datAll[i]
            lab = labAll[i]
            clustLab = clustLabAll[i]
    
            if (aCount==0 and precomputed==False and useGtCP == False): # first time through or non precomputed metrics
                # Use ground truth changepoints
#                   (wStatStore[i], wStatPks, partitionBarycenters, barycenterDistance) = wStatPartitioning(dat, "tmp", window, stride, nBack, th, outPath = 'tmp', wStat = wStatStore[i] )
#                gtChangePoints = np.argwhere(lab==1)-window
                (w2Samp,w2SampC, w2PkIdx, w2Assign, pBary_p, pBary_w) =  w2SampClustering(datOAll[i], "tmp", window, stride, nClust, w2ConvFilter, th = alphaVecW2[aCount])
                w2SampAll.append(w2Samp[window:-window])
                #Recompute for size compatabilities
                (w2Samp,w2SampC, w2PkIdx, w2Assign, pBary_p, pBary_w) =  w2SampClustering(dat, "tmp", window, stride, nClust, w2ConvFilter, th = alphaVecW2[aCount], w2Samp = w2SampAll[i])

            else: # Second time through or precomputed
                (w2Samp,w2SampC, w2PkIdx, w2Assign, pBary_p, pBary_w) =  w2SampClustering(dat, "tmp", window, stride, nClust, w2ConvFilter, th =  alphaVecW2[aCount], w2Samp = w2SampAll[i], useGtCP = useGtCP)

            #for w2samp
            w2CpOut = np.zeros(len(w2Samp))
            w2CpOut[w2PkIdx]=1
            w2CpLabel = ComputeChangePointLabels(w2PkIdx, w2Assign, len(w2Samp))
            (sampleAccuracy, map1) = LabelAccuracy11(clustLab, w2CpLabel)
            w2CpLabelT = TransformMap(w2CpLabel,map1)

            gtCpAll.append(lab)
            gtLabAll.append(clustLab)

            w2SampCAll.append(w2SampC)
            w2CpAll.append(w2CpOut)
            w2LabAll.append(w2CpLabelT)

        print("2Samp Wass Results")
        EvaluateAllMetrics(gtCpAll, w2CpAll, gtLabAll, w2LabAll, w2SampCAll, delay, alphaVecW2[aCount])

        aIdx+=1
        
