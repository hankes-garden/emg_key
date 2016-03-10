# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 19:45:42 2016

@author: jason
"""
import coding_tools as ct

import matlab.engine as mtEngine
import matlab
import numpy as np


def encode(eng, arrMsg_bin, n=7, k=4, strCoder='hamming/binary'):
    """
        encode the message with given coder
    """
    # convert to binary representation in matlab
    msg = matlab.logical(list(arrMsg_bin))
    
    encoded = eng.encode(msg, n, k, strCoder)
    
    # convert to python array
    lsEncoded = []    
    for row in encoded:
        for item in row:
            lsEncoded.append(int(item) )
    return np.array(lsEncoded)

def decode(eng, arrEncoded_bin, n=7, k=4, strCoder='hamming/binary'):
   """
       decode encrypted message via matlab FEC coders
   """
   # convert to binary representation
   encoded = matlab.double(list(arrEncoded_bin) )
   
   # decode
   decoded = eng.decode(encoded, n, k, strCoder, nargout=1)
   
   # convert to python array
   lsDecoded = []
   for row in decoded:
       for item in row:
           lsDecoded.append(int(item) )
   return np.array(lsDecoded)
   

def test(eng, arrMsg, nErrorBits, n=7, k=4, strCoder='hamming/binary'):
    arrMsg_bin = ct.toBinaryArray(arrMsg, nBitBlock=n)
    
    # encode
    arrEncoded_bin = encode(eng, arrMsg_bin, n, k, strCoder)
    
    # introduce some errors
    arrErrorPos = np.random.choice(range(len(arrEncoded_bin)), nErrorBits,
                                   replace=False)
    for i in arrErrorPos:
        arrEncoded_bin[i] = (arrEncoded_bin[i] + 1)%2
    
    # decode
    arrDecoded_bin = decode(eng, arrEncoded_bin, n, k, strCoder)
    
    print "message(%d): " % len(arrMsg_bin), arrMsg_bin
    print "encoded(%d): " % len(arrEncoded_bin), arrEncoded_bin
    print "decoded(%d): " % len(arrDecoded_bin), arrDecoded_bin
    
    # BER
    nCount = 0
    for i in xrange(len(arrMsg_bin) ):
        if (arrDecoded_bin[i] != arrMsg_bin[i]):
            nCount += 1
    dBER = nCount*1.0/len(arrMsg_bin)
    print "BER: ", dBER
    
def test_error_correction(eng, arrEncoded, nErrorBits, 
                          n=7, k=4, strCoder='hamming/binary'):
    # data1
    arrEncoded_1 = arrEncoded
    arrEncoded_bin_1 = ct.toBinaryArray(arrEncoded_1, nBitBlock=n)
    print "data1(%d): "%len(arrEncoded_bin_1), arrEncoded_bin_1

    # data2
    arrEncoded_bin_2 = np.copy(arrEncoded_bin_1)
    # introduce some errors
    arrErrorPos = np.random.choice(range(len(arrEncoded_bin_2)), nErrorBits,
                                   replace=False)
    for i in arrErrorPos:
        arrEncoded_bin_2[i] = (arrEncoded_bin_2[i] + 1)%2
    print "data2:(%d) "%len(arrEncoded_bin_1), arrEncoded_bin_2
    
    # decode
    print("coder: %s" % strCoder)
    arrDecoded_bin_1 = decode(eng, arrEncoded_bin_1, n, k, strCoder)
    arrDecoded_bin_2 = decode(eng, arrEncoded_bin_2, n, k, strCoder)
    
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
    
#%%
    print "** error correction test **"
    n = 7
    k = 4
    strCoder = 'hamming/binary'
    arrEncoded = np.random.randint(0, 2, 140)
    test_error_correction(eng, arrEncoded, 5, n, k, strCoder)
#%%
    eng.quit()
