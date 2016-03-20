# -*- coding: utf-8 -*-
"""
This script provides function to call golay coder in matlab.

Created on Mon Feb 29 17:11:19 2016

@author: jason
"""
import matlab.engine as mtEngine
import matlab
import numpy as np


def encode(eng, arrMsg_bin, k=12):
    """
        transform binary array into m*k format, then encode the data, 
        return encoded data in numpy array format
        
        Params:
        ---------
        eng: 
            the instance of matlab eng
        arrMsg_bin:
            binary-representation of message
        k:
            the number of golay code, can only be 12
            
        Returns:
        ----------
        arrEncoded_bin:
            the encrypted data in numpy array format
    """
    if( np.sum(arrMsg_bin==0) + \
        np.sum(arrMsg_bin==1) != np.size(arrMsg_bin)  ):
        raise ValueError("the data should be in binary representation.")
    
    # transform to m*k binary format
    nCols = k
    nRows = len(arrMsg_bin)/nCols
    if(nRows*nCols != np.size(arrMsg_bin) ):
        raise ValueError("the msg length should be m*k.")
        
    msg = matlab.mlarray.logical(list(arrMsg_bin), [nCols,nRows])
    msg = eng.transpose(msg) # trick to create matlab matrix in required order
    encoded_bin = eng.golaycodec(msg, True)
    
    # convert to binary array
    lsEncoded = []
    for row in encoded_bin:
        for item in row:
            lsEncoded.append(int(item) ) 
    
    return np.array(lsEncoded)
    
    
def decode(eng, arrEncoded_bin, n=23):
    """
        transform binary array to m*n format, then decode the encypted data
        
        Params:
        -------
        eng:
                matlab enginee instance
                
        arrEncoded_bin:
                encoded data in binary representation
                
        n:
            the number of golay code can only be 23
            
        Returns:
        --------
        arrDecoded_bin:
            decoded data in binary representation
        arrError_bin:
            detected error in binary representation
    """
    if( np.sum(arrEncoded_bin==0) + \
        np.sum(arrEncoded_bin==1) != np.size(arrEncoded_bin)  ):
        raise ValueError("the data should be in binary representation.")
        
    # transform to m*23 binary format
    nCols = n
    nRows = len(arrEncoded_bin)/nCols
    if(nRows*nCols != np.size(arrEncoded_bin) ):
        raise ValueError("the data length should be m*n.")
        
    encoded = matlab.mlarray.logical(list(arrEncoded_bin), [nCols,nRows])
    encoded = eng.transpose(encoded)

    # decode
    decoded = eng.golaycodec(encoded, False)

    
    # convert to binary array
    lsDecoded = []
    for row in decoded:
        for item in row:
            lsDecoded.append(int(item) )
    
    return np.array(lsDecoded)
    
def test(eng, nErrorBits=3):
    arrMsg_bin = np.random.randint(0, 2, 12)
    
    # encode
    arrEncoded_bin = encode(eng, arrMsg_bin)
    
    # introduce some error
    arrEncoded_noise = np.copy(arrEncoded_bin)
    arrErrPos = np.random.choice(len(arrEncoded_bin), nErrorBits, False)
    for pos in arrErrPos:
        arrEncoded_noise[pos] = np.bitwise_xor(arrEncoded_noise[pos], 1)
    
    # decode
    arrDecoded_bin = decode(eng, arrEncoded_bin)
    

    print "encoded: ", arrEncoded_bin
    print "noised : ", arrEncoded_noise
    print "msg    : ", arrMsg_bin
    print "decoded: ", arrDecoded_bin
    print "nErrorCount: ", \
           np.size(arrMsg_bin)-np.sum(arrMsg_bin==arrDecoded_bin)
    
if __name__ == '__main__':
    test(eng)
