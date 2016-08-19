# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 15:12:42 2016

@author: jason
"""

import numpy as np
import operator
import math


def zeroPadding(arrData, nBlockSize):
    nPadding = len(arrData)%nBlockSize
    if nPadding != 0:
        arrData = np.concatenate([arrData, 
                                  np.zeros(nBlockSize-nPadding, np.int) ] )
         
    return arrData, nPadding

def toBinaryArray(arrData, nBitNum):
    """
        convert a integer array to binary array
    """       
#    nBitNum = int(math.log(np.max(arrData), 2)+1) # largest bit
    arrBits = None
    for i in arrData:
        # [2:] to chop off the "0b" part
        arr = np.array([int(digit) for digit in bin(i)[2:]]) 
        nBitCount = len(arr)
        if (nBitCount<nBitNum): # left-zero-zeroPadding
            arr = np.concatenate((np.zeros(nBitNum-nBitCount, 
                                           dtype=np.int), arr))
            
        arrBits = np.concatenate((arrBits, arr) ) if arrBits is not None \
                    else arr
    return arrBits
    

def interleave(arrData_bin, nBuckets=7):
    """
        interleave data in a block way
    """
    if(nBuckets >= len(arrData_bin) ):
        raise ValueError("bucket number should less than data length.")

    arrData_bin, nPadding = zeroPadding(arrData_bin, nBuckets)   
                  
    nRows = len(arrData_bin)/nBuckets
    narrData = np.reshape(arrData_bin, (nRows, nBuckets) )
    
    return np.ravel(np.flipud(narrData), order='F'), nPadding
           
    
def computeBER(arrCode1, arrCode2):
    """
        compute the BER of two code, i.e., the dismatching rate. If the data
        is not binary, then transform them first.
    """
    if(len(arrCode1) != len(arrCode2) ):
        raise ValueError("the lengths of two codes should be the same.")
    
    # convert if it is not binary
    arrCode1_bin = arrCode1
    arrCode2_bin = arrCode2
    if(all([v==0 or v==1 for v in arrCode1]) is False):
        arrCode1_bin = toBinaryArray(arrCode1, 2)
    if(all([v==0 or v==1 for v in arrCode2]) is False):   
        arrCode2_bin = toBinaryArray(arrCode2, 2)
        
    nDismatchCount = np.size(arrCode1_bin) - np.sum(arrCode1_bin==arrCode2_bin)
    dBER = nDismatchCount*1.0/np.size(arrCode1_bin)
    return nDismatchCount, dBER
    

    
if __name__ == '__main__':
    a = np.arange(12)
    
    nBuckets = 3
    b, nPadding = interleave(a, nBuckets)
    print a
    print b
