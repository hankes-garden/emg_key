# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:26:19 2015

@author: jason
"""

from fastdtw import fastdtw
import matplotlib.pyplot as plt
#import frechet_distance as frechet
import pdb
import pandas as pd

import numpy as np


FLAT_PATTERN_STD = 0.002

SHAPE_CODE_INCREASE = 1
SHAPE_CODE_DECREASE = 2
SHAPE_CODE_FLAT = 3
SHAPE_CODE_CONVEX = 4
SHAPE_CODE_CONCAVE = 5


def generateShapeTemplates(nLen, dRange, dPeakRatio = 1.0):
    """
        generate 4 templates of basic shapes
    """
    dMin = 0.0
    dMax = dMin + dRange
    arrFlat = np.array([dMin,]*nLen)
    arrIncrease = np.array([dMin+i*dRange/nLen for i in xrange(nLen)] )
    arrDecrease = np.array([dMax-i*dRange/nLen for i in xrange(nLen)] )
    lsConcave = [dMin+dRange*dPeakRatio - dPeakRatio*i*2.0*dRange/nLen \
                 for i in xrange(nLen/2)] + \
                [dMin + dPeakRatio*i*2.0*dRange/nLen \
                 for i in xrange(nLen/2)]
    if (len(lsConcave) < nLen):
        lsConcave.append(dMin+dRange*dPeakRatio)
    arrConcave = np.array(lsConcave)
    
    lsConvex = [dMin + dPeakRatio*i*2.0*dRange/nLen \
                for i in xrange(nLen/2)] + \
               [dMin+dRange*dPeakRatio - dPeakRatio*i*2.0*dRange/nLen \
                for i in xrange(nLen/2)]
    if (len(lsConvex) < nLen ):
        lsConvex.append(dMin)
    arrConvex = np.array(lsConvex)
    
    return {SHAPE_CODE_FLAT: arrFlat,
            SHAPE_CODE_INCREASE: arrIncrease, 
            SHAPE_CODE_DECREASE: arrDecrease}
            
    
def shapeEncoding(arrData_raw, nCodingWndSize, nNeighbors=3):
    '''
        souce encoding according to signal shape
        
        Parameters:
        -----------
        arrData : 
                    data
        nCodingWndSize : 
                        coding window size in number of data points
        nNeighbors : 
                    the number of neighbor window to determine the data range,
                    this value should be an odd number
        
        Returns:
        --------
        lsCode : 
                a list of integer codes
        arrDataShape : 
                        the numpy.array of approximating shapes
        
    '''
    lsDataCode = []
    arrDataShape = None
    arrWndShape = None
    nDataLen = len(arrData_raw)
    arrData = arrData_raw / np.max(arrData_raw)
    
    for nStartIndex in xrange(0, nDataLen, nCodingWndSize):
        nEndIndex = nStartIndex + nCodingWndSize
        if(nEndIndex > nDataLen):
            # forget about the last segment if it is shorter than nWndSize
            break
        
        arrWndData = arrData[nStartIndex: nEndIndex]
        arrWndData_shift = arrWndData - np.min(arrWndData) # remove base line
        
#        # find nearby windows
#        nNeighborStart, nNeighborEnd = None, None
#        if (nStartIndex-(nNeighbors-1)/2*nCodingWndSize <= 0):
#            nNeighborStart = 0
#            nNeighborEnd = min(len(arrData), 
#                               nNeighborStart + nNeighbors*nCodingWndSize)
#        elif (nEndIndex + (nNeighbors-1)/2*nCodingWndSize >= nDataLen):
#            nNeighborEnd = len(arrData)
#            nNeighborStart = max(0, nNeighborEnd-nNeighbors*nCodingWndSize)
#        else:
#            nNeighborStart = max(0,\
#                                 nStartIndex-(nNeighbors-1)/2*nCodingWndSize)
#            nNeighborEnd = min(len(arrData), 
#                               nNeighborStart + nNeighbors*nCodingWndSize)
#        # compute the max range of neighbors    
#        dMaxNeighborRange = 0.0
#        for s in xrange(nNeighborStart, nNeighborEnd, nCodingWndSize):
#            dRange = np.ptp(arrData[s:s+nCodingWndSize])
#            if(dRange >= dMaxNeighborRange):
#                dMaxNeighborRange = dRange
            
        # generate patterns
        dcPatterns = generateShapeTemplates(nCodingWndSize, 
                                            np.ptp(arrWndData_shift) )
        
        # examine the shape of data within window
        nDebugIndex = None
        if (nDebugIndex is not None and \
            nStartIndex <= nDebugIndex and nEndIndex > nDebugIndex):
            pdb.set_trace()        
        
        # find nearest pattern
        nCode = None
        if(np.std(arrWndData_shift) <= FLAT_PATTERN_STD ):
            nCode = SHAPE_CODE_FLAT
            arrWndShape = np.zeros(nCodingWndSize)
        else:
            dCriteria = float("inf")
            for i, arrShape in dcPatterns.iteritems():
                dDis, path = fastdtw(arrWndData_shift, arrShape,
                                 dist=lambda a, b: abs(a-b)**2.0 )
                if(dDis < dCriteria):
                    dCriteria = dDis
                    nCode = i
                    arrWndShape = arrShape / np.max(arrShape)
                    
        # update arrDataShape and code list
        arrDataShape = arrWndShape if arrDataShape is None \
            else np.concatenate([arrDataShape, arrWndShape])
        lsDataCode.append(nCode)
    return lsDataCode, arrDataShape
    
    
if __name__ == "__main__":
    dSamplingFreq = 100.0
    arrTime = np.linspace(0.01, 2.0, 2*dSamplingFreq)

    f = 5 + arrTime
    A = 30 
    
    arr = A * np.sin(2.0*np.pi*f*arrTime) + 30


    nCodingWndSize=7
    lsCode, arrShape = shapeEncoding(arr, nCodingWndSize)
    plt.figure()
    plt.plot(arr, color='b', lw=2, alpha=0.5)
    plt.plot(arrShape, color='r')
    for i in xrange(0, len(arr), nCodingWndSize):
        plt.axvline(i, ls='-.', color='k')
    plt.show()
    print lsCode
