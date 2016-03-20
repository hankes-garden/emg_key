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

from lshash import LSHash
import fec_wrapper as fec
import golayCodeWrapper as golay
import RSCodeWrapper as rs

import math
import operator
import matlab.engine as matEngine
import matlab
import numpy as np

CODER_GOLAY = 'golay23'
CODER_HAMMING = 'hamming/binary'
CODER_RS = 'reed_solomon'

def repetitiveDecode(arrEncoded, k):
    """
        error correction via repetive coding
    """
    # zeroPadding
    arrEncoded, nPadding = ct.zeroPadding(arrEncoded, k)
    
    # decode
    lsDecoded = []
    for nStart in xrange(0, len(arrEncoded), k):
        arrBin = arrEncoded[nStart: nStart+k]
        dcValueCount = {}
        for i in arrBin:
            nCount = dcValueCount.get(i, 0)
            dcValueCount[i] = nCount + 1
        nCodeWord = max(dcValueCount.iteritems(), 
                         key=operator.itemgetter(1))[0]
        lsDecoded.append(nCodeWord)
    
    return np.array(lsDecoded)

def lshDecode(arrData_bin, lsh, n, k):
    """
    """
    lsDecoded = []
    for i in xrange(0, len(arrData_bin), n):
        arrMsg = arrData_bin[i: i+n]
        strKey = lsh.index(arrMsg)[0]
        arrKey = np.array([int(c) for c in strKey])
        lsDecoded.append(arrKey)
    arrDecoded = np.concatenate(lsDecoded)
    return arrDecoded
    
def computeDelta(matlabEng, arrData_bin_1, n, k, m, strCoder):
    """
        compute the delta for reconciliation
    """
    # find corresponding codeword of data 1
    arrMsg_bin_1 = None
    arrCodeword_bin_1 = None

    if(strCoder == CODER_GOLAY):
        arrMsg_bin_1 = golay.decode(matlabEng, arrData_bin_1, n)
        arrCodeword_bin_1 = golay.encode(matlabEng, arrMsg_bin_1, k)
    elif (strCoder == CODER_RS):
        arrMsg_bin_1 = rs.decode(matlabEng, arrData_bin_1, n, k, m)
        arrCodeword_bin_1 = rs.encode(matlabEng, arrMsg_bin_1, n, k, m)
    elif(strCoder == CODER_HAMMING):
        arrMsg_bin_1 = fec.decode(matlabEng, arrData_bin_1, 
                                  n, k, strCoder)
        arrCodeword_bin_1 = fec.encode(matlabEng, arrMsg_bin_1,
                                       n, k, strCoder)
    else:
        raise ValueError("Unkown coder")
        
    # compute the difference
    arrDelta = np.bitwise_xor(arrData_bin_1, arrCodeword_bin_1)
    
    return arrDelta
    
def reconciliate(matlabEng, arrDelta, arrData_bin_2, n, k, m, strCoder):
    """
        Given the delta, try to deduce data1 via reconciliation
    """
    arrMsg_bin_2 = None
    arrCodeword_bin_2 = None    
    
    if (strCoder == CODER_GOLAY):
        arrMsg_bin_2 = golay.decode(matlabEng, 
                                    np.bitwise_xor(arrData_bin_2, arrDelta),
                                    n)
        arrCodeword_bin_2 = golay.encode(matlabEng, arrMsg_bin_2, k)
        
    elif (strCoder == CODER_RS):
        arrMsg_bin_2 = rs.decode(matlabEng,
                                 np.bitwise_xor(arrData_bin_2, arrDelta),
                                 n, k, m)
        arrCodeword_bin_2 = rs.encode(matlabEng, arrMsg_bin_2, n, k, m)
        
    elif (strCoder == CODER_HAMMING):
        arrMsg_bin_2 = fec.decode(matlabEng,
                                  np.bitwise_xor(arrData_bin_2, arrDelta),
                                  n, k, strCoder)
        arrCodeword_bin_2 = fec.encode(matlabEng, arrMsg_bin_2, n, k)
    else:
        raise ValueError("Unkown coder")
        
    # deduce data 1 from data 2 + delta
    arrDeducedData_bin_1 = np.bitwise_xor(arrCodeword_bin_2, arrDelta)
    
    return arrDeducedData_bin_1
    
    
def test_error_correction(arrEncoded, nErrorBits):
    # data1
    arrEncoded_1 = arrEncoded
    arrEncoded_bin_1 = ct.toBinaryArray(arrEncoded_1)

    # data2
    arrEncoded_bin_2 = np.copy(arrEncoded_bin_1)
    # introduce some errors
    arrErrorPos = np.random.choice(range(len(arrEncoded_bin_2)), nErrorBits,
                                   replace=False)
    for i in arrErrorPos:
        arrEncoded_bin_2[i] = (arrEncoded_bin_2[i] + 1)%2
    
    # error correction
    n = 7
    k = 4
    lsh = LSHash(k, n)
    arrDecoded_bin_1 = lshDecode(ct.zeroPadding(arrEncoded_bin_1, n)[0], 
                                 lsh, n, k)
    arrDecoded_bin_2 = lshDecode(ct.zeroPadding(arrEncoded_bin_2, n)[0], 
                                 lsh, n, k)
    
    
    # BER
    nErrorBits, dBER = ct.computeBER(arrEncoded_bin_1, arrEncoded_bin_2)
    nErrorBits_ec, dBER_ec = ct.computeBER(arrDecoded_bin_1, arrDecoded_bin_2)
    
    return nErrorBits, dBER, nErrorBits_ec, dBER_ec
    

def test_random_secret(matlabEng, n, k, nErrorBits=1):
    # data 1
    arrData_1 = np.random.randint(0, 2, n)
    arrData_bin_1 = ct.toBinaryArray(arrData_1)
    print "d1 (%d): "%len(arrData_bin_1), arrData_bin_1

    # data2
    arrData_bin_2 = np.copy(arrData_bin_1)
    # introduce some errors
    arrErrorPos = np.random.choice(range(len(arrData_bin_2)), nErrorBits,
                                   replace=False)
    for i in arrErrorPos:
        arrData_bin_2[i] = (arrData_bin_2[i] + 1)%2
    print "d2 (%d): "%len(arrData_bin_1), arrData_bin_2
    
    
    # create secret at random
    arrSecret = np.random.randint(0, 2, k)
    arrSecret_bin = ct.toBinaryArray(arrSecret)
    arrCodeword_bin = fec.encode(matlabEng, arrSecret_bin, 
                                 n, k, 'hamming/binary')
    
    # compute the difference
    arrDelta = np.bitwise_xor(arrData_bin_1, arrCodeword_bin)
    
    # deduce securet
    arrCodeword_bin_2 = np.bitwise_xor(arrData_bin_2, arrDelta)
    arrSecret_bin_2 = fec.decode(matlabEng, arrCodeword_bin_2,
                                 n, k, 'hamming/binary')
                                 
                                 
    # compute the BER
    nMismatchedBits = np.size(arrSecret_bin) - \
                            np.sum(arrSecret_bin==arrSecret_bin_2)
    print "error bit=", nMismatchedBits, ", BER=",\
          nMismatchedBits*1.0/np.size(arrSecret_bin)
    
def test_reconciliation(nTrials, matlabEng, nKeySize, nErrorBits=1, 
                        n=7, k=4, m=3, strCoder='hamming/binary'):
    """
        test the reconciliation method mentioned in 'proximate, mobisys'11'
    """
    lsStat = []
    for i in xrange(nTrials):        
        # data 1
        arrData_1 = np.random.randint(0, 2, nKeySize)
        arrData_bin_1 = ct.toBinaryArray(arrData_1)
    
        # data2
        arrData_bin_2 = np.copy(arrData_bin_1)
        # introduce some errors
        arrErrorPos = np.random.choice(range(len(arrData_bin_2)), nErrorBits,
                                       replace=False)
        for i in arrErrorPos:
            arrData_bin_2[i] = (arrData_bin_2[i] + 1)%2
            
        # padding
        nPaddingSize = m*n if strCoder == CODER_RS else n
        arrData_bin_1 = ct.zeroPadding(arrData_bin_1, nPaddingSize)[0]
        arrData_bin_2 = ct.zeroPadding(arrData_bin_2, nPaddingSize)[0]
              
        # compute the difference
        arrDelta = computeDelta(matlabEng, arrData_bin_1, n, k, m, strCoder)
        
        # reconciliation
        arrDeducedData_bin_1 = reconciliate(matlabEng, arrDelta,
                                            arrData_bin_2, n, k, m, strCoder)

                                     
                                     
        # compute the BER
        nMismatchedBits = np.size(arrData_bin_1) - \
                                np.sum(arrData_bin_1==arrDeducedData_bin_1)
        dBER = nMismatchedBits*1.0/np.size(arrData_bin_1)
        lsStat.append({'err_bits': nMismatchedBits, 'BER': dBER})
        
        if (nMismatchedBits != 0):
            print nMismatchedBits, arrErrorPos
        
#        print "error bit=", nMismatchedBits, ", BER=", dBER
    
    dAvgErrBits = np.mean([i['err_bits'] for i in lsStat ])  
    dAvgBER = np.mean([i['BER'] for i in lsStat ])  
    print "----\n AvgErrorBits=%.3f, BER=%.3f" % (dAvgErrBits, dAvgBER)
    

    
    
if __name__ == '__main__':
    m = 4
    n = 2**m -1
    k = 9
    r = math.ceil((n-k)/2.0)
    print "max error number = ", r
        
    
    test_reconciliation(100, eng, nKeySize=200, nErrorBits=int(r)+2, 
                        n=n, k=k, m=m, strCoder=CODER_RS)
