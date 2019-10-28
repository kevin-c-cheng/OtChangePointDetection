# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:22:45 2019

@author: Kevin Cheng
"""
import numpy as np
import random as rd
import scipy as sp
import matplotlib.pyplot as plt
    
def F_GInv(F,G):
    F=F.flatten()
    G=G.flatten()
    n = len(F)
    m = len(G)
    Fsort = np.sort(F)
    Gsort = np.sort(G)
    outX = np.zeros(m)
    outY = np.zeros(m)
    for i in range(m):
        dist = np.argwhere(Fsort <= Gsort[i])
        outY[i] = len(dist)/n
        outX[i] = (i+1)/m # cdf jumps at points in Gm
    return (outX, outY)

def DistanceToUniform(p, cdf):
    #This only computes distance, we need squared distance. 
    # we assume that [0,0], and [1,1] are not included
    p=np.append(p,1)
    cdf = np.append(cdf,1)
    prevX = 0
    prevY = 0
    total = 0
    overUnder = 0

    for i in range(len(p)):
        if (cdf[i] < p[i]): # we are under
            if overUnder == 1: # we were over
                total += (np.abs(prevX-prevY) + (p[i]-prevY))/2 * (p[i]-prevX) # trapezoid
            elif overUnder !=-1: # we stayed under
                total += (np.abs(prevX-prevY) + (cdf[i]-prevY))/2 *(p[i]-prevX)
            overUnder = 1
        elif (cdf[i] < p[i]): # we are over
            if overUnder == -1: # and now we are under
                total += (np.abs(prevX-prevY) + 0)/2 * (prevY - prevX)
                total += (0 + (p[i]-cdf[i]))/2 * (p[i] - prevY)
            elif overUnder !=-1: # we are still over
                # we need to check if we fell under for some part
                if (p[i] < prevY): # if we did we have to integrate 2 smaller triangles
                    total += (np.abs(prevX-prevY))/2 * (prevY-prevX)
                    total += (p[i]-prevY)/2 * (p[i]-prevY)
                else:
                    total += (np.abs(prevY-prevX)+(cdf[i] - p[i]))/2 * (p[i]-prevX)
            overUnder = 0
        else:
            total+= np.abs(prevY-prevX)/2*(p[i]-prevX)
        
        prevX=p[i]
        prevY = cdf[i]
    return total

def DistanceSquaredToUniform(pp,cdf,step= 0.01):
    pp=np.append(pp,1)
    cdf = np.append(cdf,1)
    xAll = np.linspace(0,1,int(1/step)+1)
    total = 0 
    for x in xAll:
        argX = np.argwhere(pp>=x)
        total += (x - cdf[argX[0]])*(x - cdf[argX[0]])*step
    return total

def TwoSampleWTest(sampA, sampB, step = None):
    lenA = len(sampA)
    lenB = len(sampB)
    (cdfX, cdfY) = F_GInv(sampA,sampB)
            
    if (step is None):
        distOut = DistanceSquaredToUniform(cdfX, cdfY)*(lenA*lenB)/(lenA+lenB)
    else:
        distOut = DistanceSquaredToUniform(cdfX, cdfY, step=step)*(lenA*lenB)/(lenA+lenB)
    return distOut

def Compute2SampleWStat(dat, window, stride):
    lenDat = len(dat)
    dim=len(dat[0])
    out = np.zeros((int(np.floor(lenDat/stride)), dim))
    count = 0
    for i in range(0,lenDat-stride,stride):
        for j in range(dim):
            if (i<window or i >= lenDat-window):
                out[count,j]=0
            else:        
                win1 = dat[i-window:i,j]
                win2 = dat[i:i+window,j]
                out[count,j] = TwoSampleWTest(win1, win2)
        count = count+1
    outSingle = np.mean(out,axis=1)
    return (outSingle, out)
