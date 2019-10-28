# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:05:24 2019

@author: Kevin Cheng
"""
import os
import glob

class DataParameters:
    nDat = 0
    window = 100
    stride = 1
    nClust = 1
    delay = 0
    precomputed=False
    files = list()    

def GetDataParameters(dataSet, dataDir):
    p = DataParameters
    if (dataSet == "Beedance"):
        p.nDat = 6
        p.window = 14
        p.stride = 1
        p.nClust = 3
        p.delay = 14
        p.precomputed = False

        p.files.append(dataDir + '\\beedance\\beedance2-1.mat')
        p.files.append(dataDir + '\\beedance\\beedance2-2.mat')
        p.files.append(dataDir + '\\beedance\\beedance2-3.mat')
        p.files.append(dataDir + '\\beedance\\beedance2-4.mat')
        p.files.append(dataDir + '\\beedance\\beedance2-5.mat')
        p.files.append(dataDir + '\\beedance\\beedance2-6.mat')
    elif (dataSet == 'HASC'):
        p.nDat = 2
        p.window = 500
        p.stride = 1
        p.nClust = 7
        p.delay = 250
        p.precomputed = False

        p.files.append(dataDir + '\\HASC2011\\person671_out.mat')
        p.files.append(dataDir + '\\HASC2011\\person672_out.mat')
    elif (dataSet == 'HASC2016'):
        p.window = 500
        p.stride = 1
        p.nClust = 6
        p.delay = 250
        p.precomputed = False
        fMax = 100
        fold = dataDir + '\\HASC-PAC2016\\'
        files = glob.glob(fold + '\\*.mat')
        count=1
        for file in files:
            p.files.append(file)
            count=count+1
            if (count>fMax):
                break
        p.nDat = len(p.files)
    elif (dataSet == 'UCR'):
        p.window = 100
        p.stride = 1
        p.nClust = 2
        p.delay = 50
        p.precomputed = False
        fMax = 100
        fold = os.getcwd()+'\\UCR\\'
        files = glob.glob(fold + '\\*.mat')
        count=1
        for file in files:
            p.files.append(file)
            count=count+1
            if (count>fMax):
                break
        p.nDat = len(p.files)
    return p


def ImportParameters(fold, window=250, stride=1, nBack=1, nClust=3, delay=250, pMatch=0.75, precomputed=False, stack = 1, fMax=1000):
    p = DataParameters
    p.window = window
    p.stride = stride
    p.nBack = nBack
    p.nClust = nClust
    p.delay = delay
    p.pMatch = pMatch
    p.precomputed = precomputed
    p.stack = stack
    fMax = fMax

    files = glob.glob(fold + '*.mat')
    count=1
    for file in files:
        p.files.append(file)
        count=count+1
        if (count>fMax):
            break
    p.nDat = len(p.files)
    return p