# -*- coding: utf-8 -*-
"""
Common used functions for data analysis

Created on Fri Mar 11 11:04:54 2016

@author: jason
"""
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import scipy.stats as stats

def findExtrema(arrData, nWndSize=20):
    arrPeak = argrelextrema(arrData, np.greater, order=nWndSize)[0]
#    arrBottom = argrelextrema(arrData, np.less, order=nWndSize)[0]
#    arrExtremaIndex = np.concatenate([arrPeak, arrBottom])
    arrExtremaIndex = arrPeak
    return np.sort(arrExtremaIndex)[::-1]
    
def computeEnvelope(arrData, nWindow, nMinPeriods=None):
    """compute upper and lower envelope for given data"""
    arrUpperEnvelope = pd.rolling_max(pd.Series(arrData), window=nWindow,
                                      min_periods=nMinPeriods, center=True)
    arrLowerEnvelope = pd.rolling_min(pd.Series(arrData), window=nWindow,
                                     min_periods=nMinPeriods, center=True)
    return arrUpperEnvelope, arrLowerEnvelope
    

def normalizedCrossCorr(arrData0, arrData1):
    """
        Compute the normalized cross correlation
    """
    arrCrossCorr = np.correlate(arrData0, arrData1, mode='full')
    dDenominator = np.sqrt(np.dot(arrData0, arrData0) * \
                           np.dot(arrData1, arrData1) )
                           
    arrNormCrossCorr = arrCrossCorr/dDenominator
    nMaxIndex = np.argmax(abs(arrNormCrossCorr) )
    dCorr = arrNormCrossCorr[nMaxIndex]
    nLag = nMaxIndex - len(arrData0)
    return dCorr, nLag, arrNormCrossCorr
    

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
    

def slidingCorrelation(arrData1, arrData2, nWndSize):
    """
        compute correlation btw two signals with sliding window
    """
    nLen = min(len(arrData1), len(arrData2) )
    
    dTotalCorr = 0.0
    for nStart in xrange(0, nLen-nWndSize, nWndSize):
        nEnd = min(nStart + nWndSize, nLen)
        dCorr, p = stats.pearsonr(arrData1[nStart: nEnd], 
                                  arrData2[nStart: nEnd])
        dTotalCorr += dCorr
    dTotalCorr = dTotalCorr * nWndSize / nLen
    return dTotalCorr