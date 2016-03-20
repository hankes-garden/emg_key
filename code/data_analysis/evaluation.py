# -*- coding: utf-8 -*-
"""
This script evaluate the performance of EMG Key.

@author: jason
"""
import single_data as sd
import error_correction_coder as ecc
from tools import common_function as cf

import numpy as np
import pandas as pd
import matlab.engine

    
def evaluateReconciliationParams(lsFileList, eng):
    """
        Evaluate the effect of reconciliation parameters, i.e., m, k, on
        system performance
        
        Paramters:
        ---------
            lsFileList:
                        list of data files
            eng:
                    instance of matlab engine
    """
    # evaluate
    lsResult = []
    strCoder = ecc.CODER_RS
    for m in xrange(3,9):
        for r in xrange(1, 13):
            n = 2**m - 1
            k = n - 2*r
            if(k<=0):
                break
            
            print "testing m=%d, r=%d..." % (m, r)
            for fn in lsFileList:
                dcDataResult = sd.evaluateSingleData(strWorkingDir, fn, 
                    eng=eng, strCoder=strCoder, n=n, k=k, m=m, r=r)
                lsResult.append(dcDataResult)

    # result
    dfCurrentConf = pd.DataFrame(lsResult)
    return dfCurrentConf
    
            
    
if __name__ == '__main__':
    if('eng' not in globals() ):
        print "start matlab engine..."
        eng = matlab.engine.start_matlab()
    else:
        print "matlab engine is already existed."
    
    # select data
    strWorkingDir = "../../data/feasibility/with_attacker/"
    strFileNamePattern='yl_ww_'
    lsFileList = cf.getFileList(strWorkingDir, strFileNamePattern)
    
    # evaluate                          
    dfResult = evaluateReconciliationParams(lsFileList, eng)
    
    
    print dfResult
    