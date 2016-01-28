# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:26:19 2015

@author: jason
"""
import numpy as np
from fastdtw import fastdtw
from scipy import stats
import matplotlib.pyplot as plt
import frechet_distance as frechet
import math

def generatePattern(nLen, dMin, dMax):
    """
        generate 5 types of shape basics
    """
    dRange = dMax - dMin
    arrIncresing = np.array([dMin+i*dRange/nLen for i in xrange(nLen)] )
    arrDecresing = np.array([dMax-i*dRange/nLen for i in xrange(nLen)] )
#    arrFlat = np.array([dRange/2.0, ]*nLen)+dMin
    arrConcave = np.concatenate(\
        [np.array([dMax-i*2.0*dRange/nLen for i in xrange(nLen/2)]),
         np.array([dMin+i*2.0*dRange/nLen for i in xrange(nLen/2)])])
    arrConvex = np.concatenate(\
        [np.array([dMin+i*2.0*dRange/nLen for i in xrange(nLen/2)]),
         np.array([dMax-i*2.0*dRange/nLen for i in xrange(nLen/2)])])
    return [arrIncresing, arrDecresing, arrConcave, arrConvex]
            
def shapeEncoding(arrData, nCodingWndSize):
    '''
        souce encoding according to signal shape
    '''
    lsCode = []
    nDataLen=len(arrData)
    for nStartIndex in xrange(0, nDataLen, nCodingWndSize):
        nEndIndex = nStartIndex + nCodingWndSize
        if(nEndIndex > nDataLen):
            # forget about the last segment if it is shorter than nWndSize
            break
        
        arrSeg = arrData[nStartIndex:nEndIndex]
        
        if(np.max(arrSeg)-np.min(arrSeg)<=2.0):
            nCode = 0
        else:
            # examine the shape of segment
            lsTemplates = generatePattern(nCodingWndSize, 
                                          min(arrSeg), max(arrSeg) )
            dCriteria = 9999999999.0
            nCode = None
            for i, arrShape in enumerate(lsTemplates, 1):
                dDis, path = fastdtw(arrSeg, arrShape)
#                dDis = frechet.frechetDist(zip(range(len(arrSeg)), arrSeg),
#                                       zip(range(len(arrShape)), arrShape) )
                if(dDis < dCriteria):
                    dCriteria = dDis
                    nCode = i
        lsCode.append(nCode)
    return lsCode
    
def test():
    arr = np.random.rand(100)
    lsCode = shapeEncoding(arr, nCodingWndSize=5)
    print lsCode
    plt.plot(arr, color='b')
    
if __name__ == "__main__":
    test()