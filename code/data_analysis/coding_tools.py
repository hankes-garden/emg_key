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

def toBinaryArray(arrData, nBitBlock=1):
    """
        convert a integer array to binary array
    """
    if(np.max(arrData) <=1 and np.min(arrData)==0 ):
        return arrData
        
    nBitNum = int(math.log(np.max(arrData), 2)+1) # largest bit
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
    if (len(arrBits)%nBitBlock != 0 ):
        nPaddingSize = int(nBitBlock - len(arrBits)%nBitBlock)
        arrBits = np.concatenate((arrBits, np.zeros(nPaddingSize) ))
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
    
    return np.ravel(narrData, order='F'), nPadding
    
def deinterleave(arrData_bin, nPadding, nBuckets=2):
    """
        interleave data in a block way
    """
    if(nBuckets >= len(arrData_bin) ):
        raise ValueError("bucket number should less than data length.")
    
    nRows = int(math.ceil(len(arrData_bin)*1.0/nBuckets))
    narrData = np.reshape(arrData_bin, (nBuckets, nRows) )
    
    return np.ravel(narrData, order='F')[:-nPadding]
        
    
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
    if(max(max(arrCode1), max(arrCode2) ) != 1 or \
       min(min(arrCode1), min(arrCode2) ) != 0 ):
        arrCode1_bin = toBinaryArray(arrCode1)
        arrCode2_bin = toBinaryArray(arrCode2)
    
        
    nDismatchCount = np.size(arrCode1_bin) - np.sum(arrCode1_bin==arrCode2_bin)
    dBER = nDismatchCount*1.0/np.size(arrCode1_bin)
    return nDismatchCount, dBER
    

    
if __name__ == '__main__':
    a = np.arange(12)
    
    nBuckets = 7
    b, nPadding = interleave(a, nBuckets)
    a1 = deinterleave(b, nPadding, nBuckets)
    print a
    print b
    print a1
