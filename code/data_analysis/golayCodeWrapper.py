# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 17:11:19 2016

@author: jason
"""
import coding_tools as ct

import matlab.engine as mtEngine
import matlab
import numpy as np
import math


def encode(eng, arrMsg_bin, n=12.0):
    """
        transform binary array to m*n format, then encode the data, return
        encrypted data in matlab datatype
        
        Params:
        ---------
        eng: 
            the instance of matlab eng
        arrMsg_bin:
            binary-representation of message
        n:
            the number of golay code, can only be 23 or 24
            
        Returns:
        ----------
        encoded_bin:
            the encrypted data in matlab datatype format
    """
    
    # transform to m*23 binary format
    nCols = int(n)
    nRows = int(math.ceil(len(arrMsg_bin)*1.0/nCols))
    arrMsg_bin = np.concatenate((arrMsg_bin, 
                                np.zeros(nRows*nCols-len(arrMsg_bin) ) ) )
    msg = matlab.mlarray.logical(list(arrMsg_bin), [nCols,nRows])
    msg = eng.transpose(msg, nargout=1)
    encoded_bin = eng.golaycodec(msg, matlab.logical([1]), nargout=1)
    
    # convert to binary array
    lsEncoded = []
    for row in encoded_bin:
        for item in row:
            lsEncoded.append(int(item) ) 
    
    return np.array(lsEncoded)
    
    
def decode(eng, arrEncoded_bin, n=23.0):
    """
        transform binary array to m*n format, then decode the encypted data
        
        Params:
        -------
        eng:
                matlab enginee instance
        arrEncoded_bin:
                encoded data in binary representation
        n:
            the number of golay code (can only be 23 or 24)
            
        Returns:
        --------
        arrDecoded_bin:
            decoded data in binary representation
        arrError_bin:
            detected error in binary representation
    """
    if(np.max(arrEncoded_bin) > 1 ):
        raise ValueError("the data should be in binary representation.")
        
    # transform to m*23 binary format
    nCols = int(n)
    nRows = int(math.ceil(len(arrEncoded_bin)*1.0/nCols))
    arrEncoded_bin = np.concatenate((arrEncoded_bin, 
                                np.zeros(nRows*nCols-len(arrEncoded_bin) ) ) )
    encoded = matlab.mlarray.logical(list(arrEncoded_bin), [nCols,nRows])
    encoded = eng.transpose(encoded, nargout=1)

    # decode
    decoded_b, err_b = eng.golaycodec(encoded, matlab.logical([0]),
                                         nargout=2)

    
    # convert to binary array
    lsDecoded = []
    for row in decoded_b:
        for item in row:
            lsDecoded.append(int(item) )
            
    lsError = []
    for row in err_b:
        for item in row:
            lsError.append(int(item) )
    
    return np.array(lsDecoded), np.array(lsError)
    
def test(eng):
    arrMsg = np.arange(20)
    arrMsg_bin = ct.toBinaryArray(arrMsg)
    print "original: ", arrMsg_bin
    print "--"
    # encode
    arrEncoded_bin = encode(eng, arrMsg_bin)
    
    # decode
    arrDecoded_bin, arrError_bin = decode(eng, arrEncoded_bin)
    

    print "encoded: ", arrEncoded_bin
    print "--"
    print "decoded: ", arrDecoded_bin

def test_error_correction(eng):
    # data1
    arrEncoded_1 = np.random.randint(0, 3, 200)
    arrEncoded_bin_1 = ct.toBinaryArray(arrEncoded_1)
    print "data1(%d): "%len(arrEncoded_bin_1), arrEncoded_bin_1

    # data2
    arrEncoded_bin_2 = np.copy(arrEncoded_bin_1)
    # introduce some errors
    nErrorBits = 3
    arrErrorPos = np.random.choice(range(len(arrEncoded_bin_2)), nErrorBits,
                                   replace=False)
    for i in arrErrorPos:
        arrEncoded_bin_2[i] = (arrEncoded_bin_2[i] + 1)%2
    print "data2:(%d) "%len(arrEncoded_bin_1), arrEncoded_bin_2
    
    # decode
    arrDecoded_bin_1, arrError_bin_1 = decode(eng, arrEncoded_bin_1)
    arrDecoded_bin_2, arrError_bin_2 = decode(eng, arrEncoded_bin_2)
    print "decoded1(%d): "%len(arrDecoded_bin_1), arrDecoded_bin_1
    print "decoded2(%d): "%len(arrDecoded_bin_2), arrDecoded_bin_2
    
    # BER
    nCount = 0
    for i in xrange(len(arrDecoded_bin_1) ):
        if (arrDecoded_bin_1[i] != arrDecoded_bin_2[i]):
            nCount += 1
    dBER = nCount*1.0/len(arrDecoded_bin_1)
    print "BER: ", dBER
    
    
if __name__ == '__main__':
    eng = mtEngine.start_matlab()
    test_error_correction(eng)
    eng.quit()
