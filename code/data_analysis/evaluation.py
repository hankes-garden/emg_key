# -*- coding: utf-8 -*-
"""
This script evaluate the performance of EMG Key.

@author: jason
"""
import single_data as sd
import error_correction_coder as ecc
from tools import common_function as cf
import data_tools as dt

import numpy as np
import pandas as pd
import matlab.engine
import math

EV_SP_SET = 'ev_spec_set'
EV_RECONCILIATION = 'ev_reconciliation'
EV_BIT_RATE = 'ev_bit_rate'
EV_DISTANCE = 'ev_distance'
EV_ELETRODE_LOC = 'ev_electrode_loc'
EV_GESTURE = 'ev_gesture'
EV_MUTUAL_INFO = 'ev_mutual_info'


def evaluateReconciliationParams(eng):
    """
        Evaluate the effect of reconciliation parameters, i.e., m, k, on
        system performance
        
        Paramters:
        ---------
            lsFileList:
                        list of data files
            eng:
                    instance of matlab engine
            strCoder:
                        name of coder
            lsM:
                the possible values of m
            lsR:
                the possible values of r
       Returns:
           a pandas.DataFrame consisted of all performance result
    """
    # select data
    strWorkingDir = "../../data/evaluation/reconciliation/"
    strFileNamePattern= None
    lsFilePath = cf.getFileList(strWorkingDir, strFileNamePattern)
    
    # parameter    
    strCoder = ecc.CODER_RS
    lsM = range(4, 8)
    lsR = range(8, 20)
    dRectWnd = 2.0
    dSMWnd = 2.0
    dSCWnd = 0.15
    
    # evaluate
    lsResult = []
    if (strCoder == ecc.CODER_RS):    
        for m in lsM:
            for r in lsR:
                n = 2**m - 1
                k = n - 2*r
                if(k<1 or n*m>=500):
                    break
                
                print "testing m=%d, r=%d..." % (m, r)
                for fn in lsFilePath:
                    dcDataResult = sd.evaluateSingleData(strWorkingDir, fn,
                        dRectDuration=dRectWnd, dSMDuration=dSMWnd,
                        dSCDuration=dSCWnd,
                        eng=eng, strCoder=strCoder, n=n, k=k, m=m, r=r)
                    lsResult.append(dcDataResult)
    elif strCoder == ecc.CODER_GOLAY:
        n = 23
        k = 12
        m = 1
        r = 2
        for fn in lsFilePath:
            dcDataResult = sd.evaluateSingleData(strWorkingDir, fn, 
                dRectDuration=dRectWnd, dSMDuration=dSMWnd,
                dSCDuration=dSCWnd,
                eng=eng, strCoder=strCoder, n=n, k=k, m=m, r=r)
            lsResult.append(dcDataResult)

    # result
    dfResult = pd.DataFrame(lsResult)
    gp = dfResult.groupby([dfResult[sd.N], dfResult[sd.K]])
    dfSummary = gp.mean()
    return dfSummary, dfResult
    
def evaluateShapeCodingParams(eng):
    """
        This function evaluate the parameter of shape coding
    """
    # select data
    strWorkingDir = "../../data/evaluation/BER/"
    strFileNamePattern= None
    lsFilePath = cf.getFileList(strWorkingDir, strFileNamePattern)
    
    # params
    lsSCWnd = np.arange(0.05, 0.3, 0.05)
    dRectWnd = 2.0
    dSMWnd = 2.0
    strCoder = ecc.CODER_GOLAY
    m = 1
    n = 23
    k = 12
    r = 2
    nInterleaving = 25
    print "%s: n=%d, k=%d, m=%d, r=%d, interleave=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    # test
    lsResult = []
    for dCodingWnd in lsSCWnd:
        print "evalauting SCWnd=%.2f..." % dCodingWnd
        for fn in lsFilePath:
            dcDataResult = sd.evaluateSingleData(strWorkingDir, fn,
                                 dRectDuration = dRectWnd,
                                 dSMDurction = dSMWnd,
                                 dSCDuration = dCodingWnd,
                                 eng=eng, strCoder=strCoder, 
                                 n=n, k=k, m=m, r=r,
                                 nInterleaving=nInterleaving)
            lsResult.append(dcDataResult)
    dfResult = pd.DataFrame(lsResult)
    gp = dfResult.groupby(dfResult[sd.WND_SC])
    dfMean = gp.mean()
    return dfMean, dfResult
    
def evaluateDataSet(strLabel, strWorkingDir, lsFilePath, 
                    dRectDuration=2.0, dSMDuration=2.0, dSCDuration=0.1, 
                    eng=None, strCoder=None, n=23, k=12, m=1, r=3,
                    nInterleaving=25,
                    bSourceEncoding=True, bReconciliation=True,
                    bOutputData=False, lsOutputData=None):
    """
        Given the parameter values, this function evaluates performance
        on a specific data set.
    """
    lsResult = []
    for fn in lsFilePath:
        dcDataResult = sd.evaluateSingleData(strWorkingDir, fn,
                             dRectDuration = dRectDuration,
                             dSCDuration = dSCDuration,
                             dSMDuration = dSMDuration,
                             eng=eng, strCoder=strCoder, 
                             n=n, k=k, m=m, r=r, nInterleaving=nInterleaving,
                             bSourceEncoding=bSourceEncoding,
                             bReconciliation=bReconciliation,
                             bOutputaData = bOutputData,
                             lsOutputData = lsOutputData)
        lsResult.append(dcDataResult)
    dfResult = pd.DataFrame(lsResult)
    srMean = dfResult.mean()
    srMean.name = strLabel+"_mean"
    srStd = dfResult.std()
    srStd.name = strLabel+"_std"
    return srMean, srStd, dfResult
    
def evaluateDistance(eng):
    """
        This function evaluate the effect of distance btw A and B
    """
    # parameters
    dRectWnd = 2.0
    dSMWnd = 2.0
    dSCWnd = 0.1
    strCoder = ecc.CODER_GOLAY
    m = 1    
    n = 23
    k = 12
    r = int(math.floor((n-k)/2.0) )
    nInterleaving = 25
    print "%s: n=%d, k=%d, m=%d, r=%d, interleave=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    lsResult = []
    strWorkingDir = "../../data/evaluation/distance/"
    for strLabel in ['d1', 'd2', 'd3']:
        strFileNamePattern= strLabel
        lsFilePath = cf.getFileList(strWorkingDir, strFileNamePattern)
        
        srMean, srStd, dfDetailed = evaluateDataSet(strLabel, 
                                             strWorkingDir, lsFilePath,
                                             dRectWnd, dSMWnd, dSCWnd,
                                             eng, strCoder, n, k, m, r,
                                             nInterleaving)
        lsResult.append(srMean)
        
        # print out rotten apples
        dfSelected = dfDetailed[dfDetailed[sd.BER_USER_SRC]>=0.1]
        if(dfSelected.shape[0] != 0):
            print "--records with high BER--"
            print dfSelected[\
                [sd.FILE_NAME, sd.BER_USER_SRC, sd.BER_USER_EC]]
            print "----\n"
                
    dfSummary = pd.concat(lsResult, axis=1)
    return dfSummary
    

def evaluateEletrodeLocation(eng):
    """
        This function evaluates the installation location of electrodes
    """
    # parameters
    dRectWnd = 2.0
    dSMWnd = 2.0
    dSCWnd = 0.1
    strCoder = ecc.CODER_GOLAY
    m = 1
    n = 23
    k = 12
    r = 3
    nInterleaving = 20
    print "%s: n=%d, k=%d, m=%d, r=%d, interleave=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    lsResult = []
    strWorkingDir = "../../data/evaluation/electrode_location/"
    for strLabel in ['c1', 'c2', 'c3']:
        strFileNamePattern= strLabel
        lsFilePath = cf.getFileList(strWorkingDir, strFileNamePattern)
        if (len(lsFilePath) != 0 ):
            srMean, srStd, dfDetailed = evaluateDataSet(strLabel, 
                                                 strWorkingDir, lsFilePath,
                                                 dRectWnd, dSMWnd, dSCWnd,
                                                 eng, strCoder, 
                                                 n, k, m, r, nInterleaving)
            lsResult.append(srMean)
                                                 
            # print out rotten apples
            dfSelected = dfDetailed[dfDetailed[sd.BER_USER_SRC]>=0.1]
            if(dfSelected.shape[0] != 0):
                print "--records with high BER--"
                print dfSelected[\
                    [sd.FILE_NAME, sd.BER_USER_SRC, sd.BER_USER_EC]]
                print "----\n"
                
    dfSummary = pd.concat(lsResult, axis=1)
    return dfSummary
    
def evaluateGesture(eng):
    """
        This function evaluates the performance of different gestures.
    """
    # parameters
    dRectWnd = 2.0
    dSMWnd = 2.0
    dSCWnd = 0.1
    strCoder = ecc.CODER_GOLAY
    m = 1
    n = 23
    k = 12
    r = 3
    nInterleaving = 25
    print "%s: n=%d, k=%d, m=%d, r=%d, interleave=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    lsResult = []
    strWorkingDir = "../../data/evaluation/gesture/"
    for strLabel in ['g1', 'g2', 'g3']:
        strFileNamePattern= strLabel
        lsFilePath = cf.getFileList(strWorkingDir, strFileNamePattern)
        if (len(lsFilePath) != 0 ):
            srMean, srStd, dfDetailed = evaluateDataSet(strLabel, 
                                                 strWorkingDir, lsFilePath,
                                                 dRectWnd, dSMWnd, dSCWnd,
                                                 eng, strCoder, n, k, m, r,
                                                 nInterleaving)
                                                 
            lsResult.append(srMean)
            
            # print out rotten apples
            dfSelected = dfDetailed[dfDetailed[sd.BER_USER_SRC]>=0.1]
            if(dfSelected.shape[0] != 0):
                print "--records with high BER--"
                print dfSelected[\
                    [sd.FILE_NAME, sd.BER_USER_SRC, sd.BER_USER_EC]]
                print "----\n"
                
    dfSummary = pd.concat(lsResult, axis=1)
    return dfSummary
    
def evaluateSpecificDataSet(eng):
    # evaluate data set
    dRectWnd = 2.0
    dSMWnd = 2.0
    dSCWnd = 0.1
    strCoder = ecc.CODER_RS
    m = 4
    n = 2**m-1
    k = 3
    r = 6
    nInterleaving = 25
    print "%s: n=%d, k=%d, m=%d, r=%d, interleave=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    strWorkingDir = "../../data/evaluation/selected_set/"
    lsFilePath = cf.getFileList(strWorkingDir, None)

    srMean, srStd, dfDetailed = evaluateDataSet('selected', 
                                         strWorkingDir, lsFilePath,
                                         dRectWnd, dSMWnd, dSCWnd,
                                         eng, strCoder, n, k, m, r,
                                         nInterleaving, bOutputData=False)
    return srMean, srStd, dfDetailed
                                         
def evaluateMutualInformation():
    """
       scatter plot & mutual information 
    """
    strWorkingDir = "../../data/evaluation/mutual_info/"
    lsFilePath = cf.getFileList(strWorkingDir, None)

    lsOutputData = []
    srMean, srStd, dfDetailed = evaluateDataSet('selected', 
                                         strWorkingDir, lsFilePath,
                                         bSourceEncoding = True,
                                         bReconciliation = False,
                                         bOutputData=True,
                                         lsOutputData=lsOutputData)
    dfData = pd.concat(lsOutputData, axis=0)
    return dfData, lsOutputData

    
if __name__ == '__main__':
    if('eng' not in globals() ):
        print "start matlab engine..."
        eng = matlab.engine.start_matlab()
    else:
        print "matlab engine is already existed."
    
    strTarget = EV_SP_SET
    dfResult = None
    
    if (strTarget == EV_SP_SET): # specific set
        srMean, srStd, dfDetailed = evaluateSpecificDataSet(eng)
        print srMean
        print srStd
        
    elif (strTarget == EV_BIT_RATE): # bit rate
        pass
    
    elif (strTarget == EV_DISTANCE): # distance
        dfResult = evaluateDistance(eng)
    
    elif (strTarget == EV_ELETRODE_LOC): # electrode location
        dfResult = evaluateEletrodeLocation(eng)
        
    elif (strTarget == EV_GESTURE): # gesture
        dfResult = evaluateGesture(eng)
        
    elif (strTarget == EV_RECONCILIATION): # reconciliation
        pass
    elif (strTarget == EV_MUTUAL_INFO): # mutual info.
        dfResult, lsOutput = evaluateMutualInformation()
    else:
        raise ValueError("Unkonw evaluation target: %s." % strTarget)
    
    
