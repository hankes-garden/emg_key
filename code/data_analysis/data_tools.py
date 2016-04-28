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
import math 

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
    

def computeMI(x, y, bins, arrRange=None):
    c_xy = np.histogram2d(x, y, bins, arrRange)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi/np.log(2)
    
def computeMutualInfo(arrX, arrY):
    srX = pd.Series(arrX, name='X')
    srY = pd.Series(arrY, name='Y')
    df = pd.DataFrame([srX, srY]).T
    
    dPr_x_0 = srX.value_counts()[0] * 1.0 / srX.size
    dPr_x_1 = srX.value_counts()[1] * 1.0 / srX.size
    dPr_y_0 = srY.value_counts()[0] * 1.0 / srY.size
    dPr_y_1 = srY.value_counts()[1] * 1.0 / srY.size
    
    dPr_x0_y0 = ( (df['X']==0) & (df['Y']==0) ).sum() *1.0 / srX.size
    dPr_x0_y1 = ( (df['X']==0) & (df['Y']==1) ).sum() *1.0 / srX.size
    dPr_x1_y0 = ( (df['X']==1) & (df['Y']==0) ).sum() *1.0 / srX.size
    dPr_x1_y1 = ( (df['X']==1) & (df['Y']==1) ).sum() *1.0 / srX.size
    
    dMI = dPr_x0_y0 * math.log( dPr_x0_y0/(dPr_x_0*dPr_y_0) , 2) + \
          dPr_x0_y1 * math.log( dPr_x0_y1/(dPr_x_0*dPr_y_1) , 2) + \
          dPr_x1_y0 * math.log( dPr_x1_y0/(dPr_x_1*dPr_y_0) , 2) + \
          dPr_x1_y1 * math.log( dPr_x1_y1/(dPr_x_1*dPr_y_1) , 2)
          
    return dMI
        
    

#def computeMI(X,Y,bins):
#
#   c_XY = np.histogram2d(X,Y,bins)[0]
#   c_X = np.histogram(X,bins)[0]
#   c_Y = np.histogram(Y,bins)[0]
#
#   H_X = shannonEntropy(c_X)
#   H_Y = shannonEntropy(c_Y)
#   H_XY = shannonEntropy(c_XY)
#
#   MI = H_X + H_Y - H_XY
#   return MI

def shannonEntropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H
    
    

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
    
if __name__ == '__main__':
    a = np.random.randint(0, 500, 100)
    b = np.random.randint(0, 500, 100)
    print computeMI(a, a, 5)
    print computeMI(b, b, 5)
    print computeMI(a, b, 5)
