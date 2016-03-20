# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:58:01 2016

@author: jason
"""

# -*- coding: utf-8 -*-
"""
This script provides function to call golay coder in matlab.

Created on Mon Feb 29 17:11:19 2016

@author: jason
"""
import matlab.engine as mtEngine
import matlab
import numpy as np
import math


def encode(eng, arrMsg_bin, n, k, m):

    if( np.sum(arrMsg_bin==0) + \
        np.sum(arrMsg_bin==1) != np.size(arrMsg_bin)  ):
        raise ValueError("the data should be in binary representation.")
    
    # transform to k*m binary format
    nCols = k*m
    nRows = len(arrMsg_bin)/nCols
    if(nRows*nCols != np.size(arrMsg_bin) ):
        raise ValueError("the msg length should be a multiple times of k*m.")
        
    msg = matlab.mlarray.double(list(arrMsg_bin), [nCols,nRows])
    msg = eng.transpose(msg) # trick to create matlab matrix in required order
    encoded_bin = eng.rscodec(msg, 1, float(n), float(k), float(m) )
    
    # convert to binary array
    lsEncoded = []
    for row in encoded_bin:
        for item in row:
            lsEncoded.append(int(item) ) 
    
    return np.array(lsEncoded)
    
    
def decode(eng, arrEncoded_bin, n, k, m):
     if( np.sum(arrEncoded_bin==0) + \
        np.sum(arrEncoded_bin==1) != np.size(arrEncoded_bin)  ):
        raise ValueError("the data should be in binary representation.")
        
     # transform to n*m binary format
     nCols = n*m
     nRows = len(arrEncoded_bin)/nCols
     if(nRows*nCols != np.size(arrEncoded_bin) ):
        raise ValueError("the data length should be a multiple times of n*m.")
        
     encoded = matlab.mlarray.double(list(arrEncoded_bin), [nCols, nRows])
     encoded = eng.transpose(encoded)

     # decode
     decoded = eng.rscodec(encoded, 0, float(n), float(k), float(m) )

    
     # convert to binary array
     lsDecoded = []
     for row in decoded:
        for item in row:
            lsDecoded.append(int(item) )
    
     return np.array(lsDecoded)
    
def test(eng, nErrorBits=1):
    n = 7
    m = int(math.log(n+1, 2))
    k = 3
    
    arrMsg_bin = np.random.randint(0, 2, 4*k*m)
    
    # encode
    arrEncoded_bin = encode(eng, arrMsg_bin, n, k, m)
    
    # introduce some error
    arrEncoded_noise = np.copy(arrEncoded_bin)
    arrErrPos = np.random.choice(len(arrEncoded_bin), nErrorBits, False)
    for pos in arrErrPos:
        arrEncoded_noise[pos] = np.bitwise_xor(arrEncoded_noise[pos], 1)
    
    # decode
    arrDecoded_bin = decode(eng, arrEncoded_bin, n, k, m)
    

    print "encoded: ", arrEncoded_bin
    print "noised : ", arrEncoded_noise
    print "msg    : ", arrMsg_bin
    print "decoded: ", arrDecoded_bin
    print "nErrorCount: ", \
           np.size(arrMsg_bin)-np.sum(arrMsg_bin==arrDecoded_bin)
    
if __name__ == '__main__':
    test(eng)
