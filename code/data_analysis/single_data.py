# -*- coding: utf-8 -*-
"""
This script provides basic functions for analyzing a single data.

@author: jason
"""
import shape_encoder as encoder
import coding_tools as ct
import signal_filter as sf
import data_tools as dt
import error_correction_coder as ecc
from tools import common_function as cf

import matlab.engine
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import deque
import itertools

USER_BER_SRC = 'user_ber_src'
ATTACKER_BER_SRC = 'attacker_ber_src'

USER_BER_EC = 'user_ber_ec'
ATTACKER_BER_EC = 'attacker_ber_ec'

USER_ERR_SRC = 'user_err_src'
ATTACKER_ERR_SRC = 'attacker_err_src'

USER_ERR_EC = 'user_err_ec'
ATTACKER_ERR_EC = 'attacker_err_ec'

CODE_LEN_SRC = 'code_len_src'
CODE_LEN_EC = 'code_len_ec'

ATTACKER = 'attacker'
USER = 'user'
PAYEND = 'payend'
FILE_NAME = 'fname'
N = 'n'
M = 'm'
K = 'k'
R = 'r'
CODER = 'coder'
KN_RATIO = 'nk_ratio'


lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']


RECTIFY_ARV = 'ARV'
RECTIFY_RMS = 'RMS'

MEAN_FREQ = "mean_freq"
MEDIAN_FREQ = "median_freq"

DATA_ID_ATTACKER = 0
DATA_ID_HAND = 1
DATA_ID_PAYEND = 2

DELTA = 'delta'
DATA_1 = 'data_1'
DATA_2 = 'data_2'
DEDUCED_D1 = 'deduced_d1'
BER = 'BER'
ERROR_COUNT = 'error_count'


dcSegments = {\
'ww_20160120_220524': [(500, 3700), (5100, 8300), (10000, 12400)],
'ww_20160120_220358': [(650, 4000), (5100, 8800), (10000, 13500)],
'ww_20160120_220245': [(900, 4300), (5100, 8300), (9500, 12700)],
'ww_20160120_220126': [(750, 4600), (5800, 9500), (11100, 16000)],
'ww_20160120_220019': [(670, 3080), (4100, 7200), (8700, 12200)] }


def computeMeanFrequency(arrPSD, arrFreqIndex):
    dMeanFreq = np.sum(arrPSD.dot(arrFreqIndex))*1.0 / np.sum(arrPSD)
    return dMeanFreq
    
    
def computeMedianFrequency(arrPSD, arrFreqIndex):
    dMedianFreq = None
    dMedian = np.sum(arrPSD) / 2.0
    dSum = 0.0
    for i in xrange(len(arrFreqIndex) ):
        dSum += arrPSD[i]
        if(dSum == dMedian):
            dMedianFreq = arrFreqIndex[i]
            break
        elif(dSum < dMedian):
            pass
        elif(dSum > dMedian):
            dMedianFreq = (arrFreqIndex[i-1]+arrFreqIndex[i])/2.0
            break
        
    return dMedianFreq

def RMS(arrData):
    """
        return the root mean square of data
    """
    return math.sqrt( np.sum(arrData**2.0)/len(arrData) ) 

def rectifyEMG(arrData, dSamplingFreq, dWndDuration=0.245, method='RMS'):
    """
        change the negative value of EMG to positive values
    """
    nWndSize = int(dWndDuration * dSamplingFreq)
    arrRectified = None
    
    if (method == RECTIFY_ARV):
        arrRectified = pd.rolling_mean(np.abs(arrData), 
                                   window=nWndSize, 
                                   min_periods=1)
    elif(method == RECTIFY_RMS):
        arrRectified = pd.rolling_apply(arrData,
                                        window=nWndSize, 
                                        func=RMS,
                                        min_periods=1)
    else:
        raise ValueError("Unknown method: %s" % method)
    
    return arrRectified
    
def findEMGSegments(arrData, dSamplingFreq, 
                    dThreshold=0.9,
                    dWndDuration=0.5,
                    nQueueSize = 30,
                    dDiscontinuousDuration=2.0,
                    dMinSegmentDuration=2.0):
    """
        divides signal into segments according to std
        
    """
    return [(500, 3700), (5100, 8300), (10000, 12400)]    
    
    nWndSize = int(dWndDuration * dSamplingFreq)
    nDistance = int(math.ceil(dDiscontinuousDuration * dSamplingFreq) )
    nMinSegmentSize = int(dMinSegmentDuration*dSamplingFreq)
    arrCriteria = arrData
    dThreshold = dThreshold * np.mean(arrData)
    
    qTemp = deque()
    lsStarIndex = []
    lsEndIndex = []
    for nStart in xrange(len(arrCriteria)-nWndSize):
        nEnd = nStart + nWndSize
        dMean = np.mean(arrCriteria[nStart:nEnd])
        qTemp.append((nStart, dMean) )
        if(len(qTemp)<nQueueSize):
            continue
        elif(len(qTemp)>nQueueSize):
            qTemp.popleft()
            
        # check UPs and DOWNs
        if(np.all([i[1]<dThreshold \
            for i in itertools.islice(qTemp, 0, nQueueSize/2) ] ) \
            and np.all([i[1]>=dThreshold \
            for i in itertools.islice(qTemp, nQueueSize/2, len(qTemp)) ] )):
            # UP candidate
            nCandidate = qTemp[nQueueSize/2][0]
            lsStarIndex.append(nCandidate)
            
                
        elif(np.all([i[1]>=dThreshold \
            for i in itertools.islice(qTemp, 0, nQueueSize/2) ] ) \
            and np.all([i[1]<dThreshold \
            for i in itertools.islice(qTemp, nQueueSize/2, len(qTemp)) ] ) ):
            # DOWN candicate
            nCandidate = qTemp[nQueueSize/2][0]
            if(len(lsStarIndex)==0):
                lsStarIndex.append(300)
            lsEndIndex.append(nCandidate)
            
    nMinLen = min(len(lsStarIndex), len(lsEndIndex) )
    lsSegments = zip(lsStarIndex[:nMinLen], lsEndIndex[:nMinLen])
#    print "segment candidates: ", lsSegments
    
    # merge segments
    lsMergedSegments = []
    lsMergedSegments.append(lsSegments[0])
    for i in xrange(1, len(lsSegments) ):
        tpCurrent = lsMergedSegments[-1]
        tpNext = lsSegments[i]
        
        if(tpNext[1]-tpNext[0] <= nMinSegmentSize):
            continue # delete small segments
        
        if (tpNext[0]-tpCurrent[1]<nDistance): # merge together
            lsMergedSegments[-1] = (tpCurrent[0], tpNext[1])
        else:
            lsMergedSegments.append(tpNext)
#    print "merged segment:", lsMergedSegments, "\n-----"
    
    return lsMergedSegments
            
            
def loadData(strWorkingDir, strFileName, lsColumnNames, strFileExt = ''):
    """
        This function loads and clears a single acceleromter data

        Parameters
        ----------
        strWorkingDir: 
            working directory
        strFileName: 
            file name
        lsColumnNames: 
            a list of column names
        strFileExt: 
            file extention

        Returns
        ----------
        dfData_filtered cleared data frame

    """
    # load data
    dfData = pd.read_csv(strWorkingDir+strFileName+strFileExt,
                         index_col=0,
                         dtype=np.float32)
    dfData.columns = lsColumnNames[:len(dfData.columns)]

    # clean data
    lsMask = [True, ]* len(dfData)
    for col in dfData.columns:
        lsMask = lsMask & (dfData[col] != -1) & (~dfData[col].isnull() )
    dfData_filtered = dfData[lsMask]

    return dfData_filtered
    
     
def evaluateSingleData(strWorkingDir, strFileName, 
                       eng, strCoder, m, r, n, k,
                       bPlot=False):
    """
        Given the values of parameters, test performance on single data
    """
    dcDataResult = {}
    dcDataResult[FILE_NAME] = strFileName
    dcDataResult[M] = m
    dcDataResult[R] = r
    dcDataResult[N] = n
    dcDataResult[K] = k
    dcDataResult[KN_RATIO] = k*1.0/n
    dcDataResult[CODER] = strCoder
    
#==============================================================================
# load data
#==============================================================================
    dSamplingFreq = 240.0
    lsColumnNames = ['ch0', 'ch1', 'ch2']
    
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    # ---- setup----
    lsColumns2Inspect = ['ch0','ch1', 'ch2']
    
#==============================================================================
# process data
#==============================================================================
    # ---- raw ----
    nRawStart, nRawEnd = 0, -1
    print "--%s: [%d:%d]--" % (strFileName, nRawStart, nRawEnd)
    lsData_raw = []
    for col in lsColumns2Inspect:
        arrData = dfData[col].iloc[nRawStart: nRawEnd]
        lsData_raw.append(arrData)
        
    # ---- fft ----
    nSamples_fft = len(lsData_raw[0])
    dRes_fft = dSamplingFreq*1.0/nSamples_fft
    lsData_fft = []
    for arrData in lsData_raw:
        arrFFT = fftpack.fft(arrData)
        arrPSD = np.sqrt(abs(arrFFT)**2.0/(nSamples_fft*1.0))
        lsData_fft.append(arrPSD)
        
    # ---- filtered ----
    nLowCut, nHighCut = 5, 45
    dPowerInterference = 50.0
    nFilterOrder = 9
    nFilterShift = int(0.5 *dSamplingFreq)
    lsData_filtered = []
    for arrData in lsData_raw:
        # remove power line inteference
        arrNoise = sf.notch_filter(arrData, dPowerInterference-1., 
                                   dPowerInterference+1., 
                                   dSamplingFreq, order=nFilterOrder)
                                   
        arrFiltered = arrData - arrNoise 
        
        # remove movement artifact
        arrFiltered = sf.butter_highpass_filter(arrFiltered, cutoff=nLowCut, 
                                                fs=dSamplingFreq, 
                                                order=nFilterOrder)
        lsData_filtered.append(arrFiltered[nFilterShift:])
        
    # ---- fft on filtered ----
    nSamples_fft_filtered = len(lsData_filtered[0])
    dRes_fft_filtered = dSamplingFreq*1.0/nSamples_fft_filtered
    lsData_fft_filtered = []
    for arrFiltered in lsData_filtered:
        arrFFT = fftpack.fft(arrFiltered)
        arrPSD = np.sqrt(abs(arrFFT)**2.0/(nSamples_fft_filtered*1.0))
        lsData_fft_filtered.append(arrPSD)
        
    # ---- rectify data ----
    dRectDuration = 0.9
    nSmoothingWnd = int(1.5* dSamplingFreq)
    lsData_rectified = []
    for arrData in lsData_filtered:
        arrRect = rectifyEMG(arrData, dSamplingFreq, 
                             dWndDuration=dRectDuration,
                             method=RECTIFY_RMS)
        arrRect_sm = pd.rolling_mean(arrRect, window=nSmoothingWnd,
                                     min_periods=1)
                                     
        lsData_rectified.append(arrRect_sm)
        
    # ---- find extrema of data ----
    nExtremaWnd = int(0.5*dSamplingFreq)
    lsData_extrema = []
    for arrData in lsData_rectified:
        arrExtreInd = dt.findExtrema(arrData, nExtremaWnd)
        lsData_extrema.append(arrExtreInd)
        
    # ---- statistics of data ----
    nCodingWndSize = int(0.2*dSamplingFreq)
    dcData_stat = {}
    for i in xrange(len(lsData_rectified)):
        arrData1 = lsData_rectified[i]
        arrData2 = lsData_rectified[(i+1)%len(lsData_rectified)]
        dCorr = dt.slidingCorrelation(arrData1, arrData2,
                                       nWndSize=nCodingWndSize)
        dcData_stat["%d-%d" % (i, (i+1)%len(lsData_rectified)) ] = dCorr
        
    # ---- coding ----
    bSourceEncoding = True
    bReconciliation = True
    if(bSourceEncoding):
        # source encoding
        lsSourceCode = []           # source code in binary
        lsSourceShape = []          # shape of source
        for i, arrData in enumerate(lsData_rectified):
            lsCode, arrSourceShape = encoder.shapeEncoding(arrData,
                                                     nCodingWndSize, 5)
            arrSourceCode = np.array(lsCode)
            arrSourceCode_bin = ct.toBinaryArray(arrSourceCode)
            lsSourceCode.append(arrSourceCode_bin)
            lsSourceShape.append(arrSourceShape)
            dcDataResult[CODE_LEN_SRC] = len(arrSourceCode_bin)
            
        # reconciliation
        dcReconciliation = {}
        if(bReconciliation):
            nPadding = n*m if strCoder==ecc.CODER_RS else n
            nInterleavingSize = 10
 
            for i in [DATA_ID_HAND, DATA_ID_ATTACKER]:
                # compute delta
                arrData_bin_1 = np.copy(lsSourceCode[i])
                arrData_bin_1, nPd = ct.interleave(arrData_bin_1,
                                                   nBuckets=nInterleavingSize)
                arrData_bin_1, nPd = ct.zeroPadding(arrData_bin_1, nPadding)
                arrDelta = ecc.computeDelta(eng, arrData_bin_1, 
                                            n, k, m, strCoder)
                                            
                # reconciliation
                arrData_bin_2 = np.copy(lsSourceCode[DATA_ID_PAYEND])
                arrData_bin_2, nPd = ct.interleave(arrData_bin_2,
                                                   nBuckets=nInterleavingSize)
                arrData_bin_2, nPd = ct.zeroPadding(arrData_bin_2, nPadding)
                
                arrDeduced_bin = ecc.reconciliate(eng, arrDelta,
                                                  arrData_bin_2,
                                                  n, k, m, strCoder)
                    
                dcReconciliation[i] = {DATA_1: arrData_bin_1, 
                                       DATA_2: arrData_bin_2,
                                       DEDUCED_D1: arrDeduced_bin}
                                                
            
            # coding performance    
            for i in [DATA_ID_HAND, DATA_ID_ATTACKER]:
                # before reconciliation
                arrSrcCode1 = lsSourceCode[i]
                arrSrcCode2 = lsSourceCode[DATA_ID_PAYEND]
                nErrorBits_src, dBER_src =ct.computeBER(arrSrcCode1,
                                                        arrSrcCode2)
#                print("src: BER(%d, %d)=%.3f, error=%d, len=%d" % \
#                      (i, DATA_ID_PAYEND, dBER_src, \
#                      nErrorBits_src, len(arrSrcCode1) ) )
            
                # after reconciliation
                arrData_bin_1 = dcReconciliation[i][DATA_1]
                arrDeduced_1 = dcReconciliation[i][DEDUCED_D1]
        
                nErrorBits_recon, dBER_recon =ct.computeBER(arrData_bin_1,
                                                            arrDeduced_1)
#                print("rec: BER(%d, %d)=%.3f, error=%d, len=%d" % \
#                      (i, DATA_ID_PAYEND, dBER_recon, \
#                      nErrorBits_recon, len(arrDeduced_1) ) )
                dcDataResult[CODE_LEN_EC] = len(arrDeduced_1)
                if(i == DATA_ID_HAND):
                    dcDataResult[USER_BER_SRC] = dBER_src
                    dcDataResult[USER_BER_EC] = dBER_recon
                    dcDataResult[USER_ERR_SRC] = nErrorBits_src
                    dcDataResult[USER_ERR_EC] = nErrorBits_recon
                else:
                    dcDataResult[ATTACKER_BER_SRC] = dBER_src
                    dcDataResult[ATTACKER_BER_EC] = dBER_recon
                    dcDataResult[ATTACKER_ERR_SRC] = nErrorBits_src
                    dcDataResult[ATTACKER_ERR_EC] = nErrorBits_recon
                        
#==============================================================================
# plot
#==============================================================================
    if (bPlot):
        # look-and-feel
        nFontSize = 18
        strFontName = "Times new Roman"
        
        # plot raw 
        bPlotRawData = False
        tpYLim_raw = None
        
        # plot fft on raw data
        bPlotFFT = False
        tpYLim_fft = None
        
        # plot filtered data
        bPlotFiltered = False
        tpYLim_filtered = None
        
        # plot fft on filtered data
        bPlotFFTonFiltered = False
        tpYLim_fft_filtered = None
        
        # plot rectified data based on filtered data
        bPlotRectified = True
        bPlotAuxiliaryLine = True
        bPlotShape = True
        bAnnotateExtrema = False
        nRectShift = 50
        tpYLim_rectified = None
        
        
        # plot synchronized view
        bPlotSyncView = True
        strSyncTitle = "".join([s+"_" for s in lsColumns2Inspect] )
        
        # bAnatation
        bAnatation = True
        tpAnatationXYCoor = (.75, .9)
        
        # create axes
        nRows = np.sum([bPlotRawData, bPlotFFT, bPlotFiltered, 
                        bPlotFFTonFiltered, bPlotRectified] )
        nCols= len(lsColumns2Inspect) if bPlotSyncView is False else 1
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
        
        nCurrentRow = 0
        
        # ---- plot raw data -----
        if(bPlotRawData is True):
            for i, arrData in enumerate(lsData_raw):
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                nVerticalShift = 0 if bPlotSyncView is False else (100*i)
                axes[nRowID, nColID].plot(arrData-nVerticalShift, 
                                          color=lsRGB[i], 
                                          alpha=dAlpha,
                                          label=lsColumns2Inspect[i])
                axes[nRowID, nColID].set_xlabel(\
                    (lsColumns2Inspect[i] if bPlotSyncView is False \
                    else  strSyncTitle) + "(raw)" ) 
                if (tpYLim_raw is not None):
                    axes[nRowID, nColID].set_ylim(tpYLim_raw[0], tpYLim_raw[1])
                axes[nRowID, nColID].grid('on')
                
            nCurrentRow += 1
            
        # ---- plot FFT ----
        nDCEnd = 10
        if (bPlotFFT is True):
            for i, arrPSD in enumerate(lsData_fft):
                arrFreqIndex = np.linspace(nDCEnd*dRes_fft, 
                                           dSamplingFreq/2.0, 
                                           nSamples_fft/2-nDCEnd)
                                           
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                nVerticalShift = 0 if bPlotSyncView is False else (50*i)
                axes[nRowID, nColID].plot(arrFreqIndex,
                    arrPSD[nDCEnd:nSamples_fft/2]-nVerticalShift,
                    color=lsRGB[i], alpha=dAlpha)
                axes[nRowID, nColID].set_xticks(range(0, 
                                                int(dSamplingFreq/2), 10) )
                axes[nRowID, nColID].set_xlabel( \
                    (lsColumns2Inspect[i]  if bPlotSyncView is False \
                    else strSyncTitle ) +"(fft)" )
                if (tpYLim_fft is not None):
                    axes[nRowID, nColID].set_ylim(tpYLim_fft[0], tpYLim_fft[1])
                axes[nRowID, nColID].grid('on')
    
            nCurrentRow += 1
                
        # ---- plot filtered data ----
        if(bPlotFiltered is True):
            for i, arrFiltered in enumerate(lsData_filtered):
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                nVerticalShift = 0 if bPlotSyncView is False else (80*i)
                axes[nRowID, nColID].plot(\
                    arrFiltered-nVerticalShift,
                    color=lsRGB[i], alpha=dAlpha)
                    
                axes[nRowID, nColID].set_xlabel(\
                    (lsColumns2Inspect[i] if bPlotSyncView is False \
                    else strSyncTitle )  +"(filtered)" )
                if (tpYLim_filtered is not None):
                    axes[nRowID, nColID].set_ylim(tpYLim_filtered[0],
                                                  tpYLim_filtered[1])
                axes[nRowID, nColID].grid('on')
            nCurrentRow += 1
            
        # ---- plot FFT on filtered data ----
        if (bPlotFFTonFiltered is True):
            for i, arrPSD in enumerate(lsData_fft_filtered):
                arrFreqIndex = np.linspace(nDCEnd*dRes_fft_filtered, 
                                           dSamplingFreq/2.0, 
                                           nSamples_fft_filtered/2-nDCEnd)
                                           
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                nVerticalShift = 0 if bPlotSyncView is False else (30*i)
                
                axes[nRowID, nColID].plot(arrFreqIndex,
                    pd.rolling_mean(\
                        arrPSD[nDCEnd: int(nSamples_fft_filtered/2.0)] \
                        -nVerticalShift, window=40, min_periods=1),
                    color=lsRGB[i], alpha=dAlpha)
                    
                axes[nRowID, nColID].set_xticks(range(0, 
                                                int(dSamplingFreq/2), 10) )
                axes[nRowID, nColID].set_xlabel( \
                    (lsColumns2Inspect[i]  if bPlotSyncView is False \
                    else strSyncTitle ) +"(fft@filtered)")
                if (tpYLim_fft_filtered is not None):
                    axes[nRowID, nColID].set_ylim(tpYLim_fft_filtered[0], 
                                                  tpYLim_fft_filtered[1])
                axes[nRowID, nColID].grid('on')
            nCurrentRow += 1
        
        
        # ---- plot rectified data ----
        if(bPlotRectified is True):
            for i, arrData in enumerate(lsData_rectified):
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                nDataShift = 0 if bPlotSyncView is False else (0*i)
                nShapeShift = 0 if bPlotSyncView is False else (1*i)
                
                # rectified data
                axes[nRowID, nColID].plot(arrData+nDataShift,
                                          color=lsRGB[i], lw=3, 
                                          alpha=dAlpha)
                # approximating shapes                                      
                if(bSourceEncoding is True and bPlotShape is True):                          
                    axes[nRowID, nColID].plot(lsSourceShape[i]+nShapeShift,
                                              color=lsRGB[i],
                                              lw=2, alpha=dAlpha)
                                          
                axes[nRowID, nColID].set_xlabel(\
                    (lsColumns2Inspect[i] if bPlotSyncView is False \
                    else strSyncTitle)  +"(rect)" )
                if (tpYLim_rectified is not None):
                    axes[nRowID, nColID].set_ylim(tpYLim_rectified[0],
                                                  tpYLim_rectified[1])
                    
                # plot auxiliary line
                if(bPlotAuxiliaryLine is True):
                    #coding wnd line
                    for ln in xrange(0, len(arrData), nCodingWndSize):
                        axes[nRowID, nColID].axvline(ln, color='k', 
                                                     ls='-.', alpha=0.3)
                
                # anatation: stats
                if(bAnatation is True):
                    strKey = "%d-%d" % (i, (i+1)%len(lsData_rectified) )
                    axes[nRowID, nColID].annotate(\
                        'corr_%s = %.2f'% (strKey, dcData_stat[strKey]), 
                        xy= (.3+i*0.2, .95) if bPlotSyncView \
                            else tpAnatationXYCoor, 
                        xycoords='axes fraction',
                        horizontalalignment='center',
                        verticalalignment='center')
                        
                # anatation: extrema
                if(bAnnotateExtrema is True ):
                    arrExtremaInd = lsData_extrema[i]
                    for ind in arrExtremaInd:
                        axes[nRowID, nColID].annotate("e%d"%i, 
                            xy=(ind, arrData[ind]), 
                            xytext=(ind, arrData[ind]+3),
                            arrowprops=dict(facecolor=lsRGB[i], color=lsRGB[i],
                                            shrink=0.05,
                                            headwidth=5, width=0.5))
            nCurrentRow += 1
            
    
        fig.suptitle(strFileName, fontname=strFontName, fontsize=nFontSize)
        plt.tight_layout()
        plt.show()
        
    return dcDataResult
    
    
if __name__ == '__main__':
    if('eng' not in globals() ):
        print "start matlab engine..."
        eng = matlab.engine.start_matlab()
    else:
        print "matlab engine is already existed."
    
    # setup
    strWorkingDir = "../../data/feasibility/with_attacker/"
    strFileName = 'yl_ww_20160224_210921.txt'
    
    strCoder = ecc.CODER_RS
    m = 3    
    n = 7
    k = 3
    r = int(math.floor((n-k)/2.0) )
    
    # evaluate                          
    dcDataResult = evaluateSingleData(strWorkingDir, strFileName, eng, 
                                      strCoder, m, r, n, k, bPlot=False)
    
       
    print dcDataResult
    