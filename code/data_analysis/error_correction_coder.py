# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 20:43:15 2016

@author: jason
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 19:45:42 2016

@author: jason
"""
import coding_tools as ct

import matlab.engine as mtEngine
import matlab
import numpy as np

def decode(arrEncoded_bin):
   """
       decode encrypted message via matlab FEC coders
   """
   return arrEncoded_bin[range(0, len(arrEncoded_bin), 2 )]

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
    
def test_error_correction(arrEncoded, nErrorBits):
    # data1
    arrEncoded_1 = arrEncoded
    arrEncoded_bin_1 = ct.toBinaryArray(arrEncoded_1)
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
    arrDecoded_bin_1 = decode(arrEncoded_bin_1)
    arrDecoded_bin_2 = decode(arrEncoded_bin_2)
    
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
    arrEncoded = np.random.randint(0, 2, 100)
    test_error_correction(arrEncoded, 5)
