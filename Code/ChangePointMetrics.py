# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:40:46 2019

@author: Kevin Cheng
"""
import numpy as np
import sklearn
import scipy as sp
import random as rd
import scipy.optimize as sciop

def ChangePointMetrics_AUC(liklihood, gt):
    (fpr, tpr, thresholds) = sklearn.metrics.roc_curve(gt,liklihood, pos_label = 1)
    aucVal = sklearn.metrics.auc(fpr, tpr)
    return aucVal

def ChangePointMetrics_AUC_Full(liklihood, gt):
    (fpr, tpr, thresholds) = sklearn.metrics.roc_curve(gt,liklihood, pos_label = 1)
    aucVal = sklearn.metrics.auc(fpr, tpr)
    return (aucVal, fpr, tpr, thresholds)

def ChangePointAUCSequence(liklihood, gt):
    likAll = np.array([])
    gtAll = np.array([])
    for i in range(len(liklihood)):
#        auc = ChangePointMetrics_AUC(liklihood[i], gt[i])
#        print ("auc: " + str(np.round(auc,3)))
        likAll = np.concatenate((likAll, liklihood[i]))
        gtAll = np.concatenate((gtAll, gt[i]))
    return ChangePointMetrics_AUC_Full(likAll, gtAll)

def ChangePointMetrics_AUCpeaks(liklihood, gt, margin, nThresh=1000):
    maxN = np.max(liklihood)
    minN = np.min(liklihood)
    thresh = np.linspace(maxN, minN, nThresh)
    thresh = np.delete(thresh, 0)
    tpr= np.zeros(nThresh)
    fpr= np.zeros(nThresh)
    
    cpLoc = np.argwhere(gt==1)
    count = 0
    for th in thresh:
        (pks, props) = sp.signal.find_peaks(liklihood, height=th)
        tpr[count], fpr[count] = PeakTprFpr(pks, cpLoc, len(gt), margin)
        count = count+1
    tpr[-1]=1
    fpr[-1]=1
    return sklearn.metrics.auc(fpr, tpr)   


def PeakTprFpr(peaks, gt, nSig, margin):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if peaks.size==0:
        return 0,0
    for i in range(len(gt)): #Seach over all positive detections
        if (np.min(np.abs(gt[i]-peaks))< margin):
            TP = TP +1
        else:
            FN = FN +1
    for i in range(margin, nSig-margin): #Search over all negative 
        if (np.min(np.abs(i-gt)) >  margin): # No ground truth change point
            if (np.min(np.abs(i-peaks)) >  margin): # No detected change points
                TN = TN +1
            else: 
                FP = FP +1

    TPR = TP /(TP + FN)
    FPR = FP /(FP + TN)
    return TPR, FPR         

def DetectionDelay(peakLoc, gt):
    #assumes you have the peak locations
    cpLoc = np.argwhere(gt==1)
    delay = np.zeros(len(cpLoc))
    for i in range(len(cpLoc)):
        delay[i] = np.min(np.abs(cpLoc[i]-peakLoc))
    return delay

def DecimateGT(gt, offset, stride):
    # downsample the ground truth according to the stride parameters
    out = np.zeros(int((len(gt)-2*offset)/stride))
    for i in range(len(out)):
        out[i] = np.max(gt[offset+i*stride:offset+(i+1)*stride])
    return out

def ComputeChangePoints(labels):
    #Converts detection sample labels to change point locations
    #Returns (n-1) change point locations, and (n) partition Labels
    #does not return 0 or len(labels) as change points
    cp = np.zeros(0)
    cpLabel = np.zeros(0)
    for i in range(len(labels)-1):
        if (labels[i] != labels[i+1]):
            cp = np.append(cp, i)
            cpLabel = np.append(cpLabel, labels[i])
    cpLabel = np.append(cpLabel, labels[i+1])
    return (cp, cpLabel)

def ComputeChangePointLabels(cp, cpLabel, dataLength):
    #Convert change points and partition labels to change point sample labels
    cp = np.append(cp, dataLength)
    cp = np.append(0, cp)
    label = np.zeros(dataLength)
    for i in range(len(cp)-1):
        label[cp[i]:cp[i+1]] = np.zeros(cp[i+1]-cp[i])+cpLabel[i]
    return label

def LabelCpMode(gtLabel, cp):
    cp = np.append(cp, len(gtLabel))
    modeOut = np.zeros(len(cp))
    accOut = np.zeros(len(cp))
    start = 0
    for i in range(len(cp)):
        if (cp[i] != start): #prevents duplicates throwing erros
            res = sp.stats.mode(gtLabel[start:int(cp[i])])
            modeOut[i] = res[0][0]
            accOut[i] = res[1][0] / (int(cp[i])-start)
            start = int(cp[i])
    return (modeOut, accOut)

def ConfusionMatrix(gtLabel, label, exclude = None):
    # generate confusion matrix for ground truth (row), and sample labels (columns)
    if (len(gtLabel) != len(label)):
        print("ChangePointMetrics.ConfusionMatrix, not same size")
    u_gt = np.sort(np.unique(gtLabel))
    u_lab = np.sort(np.unique(label))
    if (exclude is not None):
        for i in range(len(exclude)):
            u_gt = np.delete(u_gt, np.argwhere(u_gt==exclude[i]))
            u_lab = np.delete(u_lab, np.argwhere(u_lab==exclude[i]))
            
    nGT = len(u_gt)
    nLabel = len(u_lab)
    
    con = np.zeros((nGT,nLabel))
    for i in range(nGT):
        for j in range(nLabel):
            con[i,j] = np.sum(np.logical_and(gtLabel==u_gt[i], label ==u_lab[j]))
    return (con / len(label), u_gt, u_lab)

def ConfusionMatrixGivenLabel(gtLabel, label, givenLabel):
    # generate confusion matrix for ground truth (row), and sample labels (columns)
    if (len(gtLabel) != len(label)):
        print("ChangePointMetrics.ConfusionMatrix, not same size")
    u_gt = givenLabel
    u_lab = givenLabel
            
    nGT = len(u_gt)
    nLabel = len(u_lab)
    
    con = np.zeros((nGT,nLabel))
    for i in range(nGT):
        for j in range(nLabel):
            con[i,j] = np.sum(np.logical_and(gtLabel==u_gt[i], label ==u_lab[j]))
    return (con / len(label), u_gt, u_lab)


def LabelAccuracy(gtLabel, label, exclude = None):
    # Find the maximum label accuracy by assigning each label to the ground truth label
    # that contributes the highest to the accuracy
    (confusion, u_gt, u_lab) = ConfusionMatrix(gtLabel, label, exclude = exclude)
    nDim = confusion.shape[1]
    labelMap = np.zeros(nDim, 2)
    tot = 0
    for i in range(nDim):
        labelMap[i,0] = u_lab[i]
        if (len(confusion[:,i])==0):
            labelMap[i,1] = -1
        else:
            labelMap[i,1] = np.argmax(confusion[:,i])
            tot = tot + np.max(confusion[:,i])
    return (tot, labelMap)

def LabelAccuracy11(gtLabel, label, exclude = None):
    # Find the maximum label accuracy by assigning each label to the ground truth label
    # that contributes the highest to the accuracy
    # we require this to be a 1-1 mapping. we use a greedy method
    (confusion, u_gt, u_lab) = ConfusionMatrix(gtLabel, label, exclude = exclude)

    # Hungarian Method
    (row_ind, col_ind) = sciop.linear_sum_assignment(1-confusion)
    labelMap=np.zeros((confusion.shape[1],2))
    labelMap[:,0]=u_gt[row_ind]
    labelMap[:,1]=u_lab[col_ind]
    
    total = confusion[row_ind, col_ind].sum()
    return (total, labelMap)


def LabelAccuracy11Sequence(gtLabel, label, exclude = None):
    totalSamp = 0
    totalAcc = 0
    totalConfusion = np.zeros((len(np.unique(gtLabel[0])),len(np.unique(gtLabel[0]))))
    for i in range(len(gtLabel)):
        (acc, labelMap) = LabelAccuracy11(gtLabel[i], label[i], exclude = exclude)
        totalAcc += acc*len(gtLabel[i])
        totalSamp+= len(gtLabel[i])
        relabel = TransformMap(label[i], labelMap)
        (con, d1, d2) = ConfusionMatrixGivenLabel(gtLabel[i],relabel, np.unique(gtLabel[i]))
        totalConfusion = totalConfusion + con*len(label[i])    
    return (totalAcc/totalSamp, totalConfusion)


def TransformMap(label, map1):
    label2=np.copy(label)
    for i in range(len(map1)):
        label2[np.argwhere(label==map1[i,1])] = map1[i,0]
    return label2

def EvaluateAccuracy(gtLabel, label, map1, exclude=None):
    tLabel = TransformMap(label, map1)
    (confusion, dump, dump1) = ConfusionMatrix(gtLabel, tLabel, exclude = exclude)
    return np.sum(np.diag(confusion))

def ChangePointF1Curve(gtLabel,label,window, nThresh=100):
    maxVal = np.max(label)
    thresh = np.linspace(0, maxVal, num=nThresh)
    F1 = np.zeros(nThresh)
    prec = np.zeros(nThresh)
    rec = np.zeros(nThresh)
    
    for i in range(nThresh):
        pks = sp.signal.find_peaks(label, height=thresh[i], distance=window/2)
        z = np.zeros(len(label))
        z[pks[0]]=1
        (F1[i], prec[i], rec[i]) = ChangePointF1Score(gtLabel, z, window)
#        (F1[i], prec[i], rec[i]) = ChangePointF1Score(gtLabel, label>=thresh[i], window)
#    auc = sklearn.metrics.auc(prec, rec)
    avgPR = sklearn.metrics.average_precision_score(gtLabel,label)
    return (F1, avgPR)

def ChangePointF1ScoreSequence(gtLabels, labels, window):
    precAll=0
    recAll = 0
    totalLab = 0
    totalGtLab = 0
    for i in range(len(gtLabels)):
        (f1,prec,rec) = ChangePointF1Score(gtLabels[i], labels[i],window)
        precAll += prec*sum(labels[i])
        recAll += rec*sum(gtLabels[i])
        totalLab += sum(labels[i])
        totalGtLab += sum(gtLabels[i])
        
    recOut = recAll/totalGtLab
    if (totalLab==0):
        precOut=0
    else:
        precOut = precAll/totalLab

    if (precOut+recOut==0):
        return (0,precOut, recOut)
    else:
        return (2*precOut*recOut/(precOut+recOut), precOut, recOut)
        
        
def ChangePointF1Score(gtLabel, label, window):
    # given ground truth sequence of labels, and real labels, computes precision, recall and f1 score assuming leniancy of (window)
    l = len(gtLabel)
    window = int(window)
    # Compute precision
    tp=0
    totalLabel = np.sum(label)
    for i in np.argwhere(label==1):
        if (np.max(gtLabel[np.maximum(1,i[0]-window):np.minimum(l,i[0]+window)]) == 1): # true change point close to label
            tp += 1
    
    fn = 0
    totalGt = np.sum(gtLabel)
    for i in np.argwhere(gtLabel==1):
        if (np.max(label[np.maximum(0,i[0]-window):np.minimum(l-1,i[0]+window)]) == 1):
            fn += 1

    precision = tp / totalLabel
    recall = fn / totalGt
    if (precision + recall == 0):
        f1=0
    else:
        f1 = 2*precision*recall/(precision+recall)
    if (totalLabel==0):
        precision=0
    if (totalGt==0):
        recall=0
    return (f1, precision, recall)


def EvaluateAllMetrics(gtCpAll, w2CpAll, gtLabAll, w2LabAll, w2StatAll, delay, alpha):
    (aucVal, fpr, tpr, thresholds) = ChangePointAUCSequence(w2StatAll,gtCpAll)
    print("  Change Point AUC: " +  str(np.round(aucVal,3)))
    (cpF1, cpPrec, cpRec) = ChangePointF1ScoreSequence(gtCpAll, w2CpAll, delay)
    print("  Change Point: alpha: " + str(np.round(alpha,3)) + " F1: " + str(np.round(cpF1,3)) + " prec: " + str(np.round(cpPrec,3)) +" rec: " + str(np.round(cpRec,3)))
    (totalAcc, tConf) = LabelAccuracy11Sequence(gtLabAll, w2LabAll, exclude=[-1])
    print("  Label Accuracy: " + str(np.round(totalAcc,3)))
    print("  Label Confusion Matrix")
    print(np.matrix(tConf/sum(sum(tConf))))
    print("")