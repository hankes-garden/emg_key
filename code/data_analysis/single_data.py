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
import scipy.stats as stats

BER_USER_SRC = 'ber_user_src'
BER_ATTACKER_SRC = 'ber_attacker_src'
BER_USER_EC = 'ber_user_ec'
BER_ATTACKER_EC = 'ber_attacker_ec'
ERR_USER_SRC = 'err_user_src'
ERR_ATTACKER_SRC = 'err_attacker_src'
ERR_USER_EC = 'err_user_ec'
ERR_ATTACKER_EC = 'err_attacker_ec'

CODE_LEN_SRC = 'code_len_src'
CODE_LEN_EC = 'code_len_ec'

ENTROPY = 'entropy'

ATTACKER = 'attacker'
USER = 'user'
PAYEND = 'payend'
FILE_NAME = 'fname'
N = 'n'
M = 'm'
K = 'k'
R = 'r'
CODER = 'coder'
KN_RATIO = 'kn_ratio'
WND_RECT = 'wnd_rect' # wnd size of EMG rectification
WND_SM = 'wnd_sm' # wnd size of smoothing
WND_SC = 'wnd_sc' # wnd size of shape coding


lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']

RECTIFY_ARV = 'ARV'
RECTIFY_RMS = 'RMS'

MEAN_FREQ = "mean_freq"
MEDIAN_FREQ = "median_freq"

DATA_ID_ATTACKER = 0
DATA_ID_USER = 1
DATA_ID_PAYEND = 2

DELTA = 'delta'
DATA_1 = 'data_1'
DATA_2 = 'data_2'
DEDUCED_D1 = 'deduced_d1'
BER = 'BER'
ERROR_COUNT = 'error_count'


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
    
def computeSymbolEntropy(arrCode):
    srValueCnt = pd.value_counts(arrCode)
    nTotal = len(arrCode)
    srProb = srValueCnt*1.0/nTotal
    dEntropy = -1.0 * \
        (srProb.multiply(srProb.apply(math.log, args=(2, ) ) ) ).sum()
    return dEntropy
    
def preprocess(dfData, lsColumns2Inspect, nRawStart, nRawEnd,
               dSamplingFreq, dSMDuration, dRectDuration):
    # raw
    lsData_raw = []
    for col in lsColumns2Inspect:
        arrData = dfData[col].iloc[nRawStart: nRawEnd]
        lsData_raw.append(arrData)
        
    # fft 
    nSamples_fft = len(lsData_raw[0])
    dRes_fft = dSamplingFreq*1.0/nSamples_fft
    lsData_fft = []
    for arrData in lsData_raw:
        arrFFT = fftpack.fft(arrData)
        arrPSD = np.sqrt(abs(arrFFT)**2.0/(nSamples_fft*1.0))
        lsData_fft.append(arrPSD)
        
    # filtered
    nLowCut = 5
    dPowerInterference = 50.0
    nFilterOrder = 9
    nFilterShift = int(0.7 *dSamplingFreq)
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
        
        
#    # ---- fft on filtered ----
#    nSamples_fft_filtered = len(lsData_filtered[0])
#    dRes_fft_filtered = dSamplingFreq*1.0/nSamples_fft_filtered
#    lsData_fft_filtered = []
#    for arrFiltered in lsData_filtered:
#        arrFFT = fftpack.fft(arrFiltered)
#        arrPSD = np.sqrt(abs(arrFFT)**2.0/(nSamples_fft_filtered*1.0))
#        lsData_fft_filtered.append(arrPSD)
        
    # rectified data
    nSmoothingWnd = int(dSMDuration* dSamplingFreq)
    lsData_rectified = []
    for arrData in lsData_filtered:
        arrRect = rectifyEMG(arrData, dSamplingFreq, 
                             dWndDuration=dRectDuration,
                             method=RECTIFY_RMS)
        arrRect_sm = pd.rolling_mean(arrRect, window=nSmoothingWnd,
                                     min_periods=1)
                                     
        lsData_rectified.append(arrRect_sm)
        
    return lsData_raw, lsData_fft, lsData_filtered, lsData_rectified
    
def plotData(lsData_raw, lsData_fft, lsData_filtered, lsData_rectified, 
             lsColumns2Inspect, dSamplingFreq,
             bPlotRawData, bPlotFFT, bPlotFiltered, bPlotRectified,
             bSourceEncoding=False, nSCWndSize=None, lsSourceShape=None, 
             bAnatation=False, dcData_stat=None):
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
    bPlotAuxiliaryLine = False
    bPlotShape = False
    nRectShift = 50
    tpYLim_rectified = None
    
    
    # plot synchronized view
    bPlotSyncView = True
    strSyncTitle = "".join([s+"_" for s in lsColumns2Inspect] )
    
    # bAnatation
    bAnatation = False
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
            axes[nRowID, nColID].plot((arrData-nVerticalShift).tolist(), 
                                      color=lsRGB[i], 
                                      alpha=dAlpha,
                                      label=lsColumns2Inspect[i])
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else  strSyncTitle) + "(raw)", fontname=strFontName,
                fontsize=nFontSize) 
            if (tpYLim_raw is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_raw[0], tpYLim_raw[1])

        # decorate the axis  
        xTickLabels = axes[nRowID, nColID].xaxis.get_ticklabels()
        plt.setp(xTickLabels, fontname=strFontName,
                     size=nFontSize)
        yTickLabels = axes[nRowID, nColID].yaxis.get_ticklabels()
        plt.setp(yTickLabels, fontname=strFontName,
                     size=nFontSize)
            
        nCurrentRow += 1
        
    # ---- plot FFT ----
    nDCEnd = 10
    nSamples_fft = len(lsData_raw[0])
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
        
    # ---- plot rectified data ----
    if(bPlotRectified is True):
        for i, arrData in enumerate(lsData_rectified):
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nDataShift = 0 if bPlotSyncView is False else (0*i)
            nShapeShift = 5 if bPlotSyncView is False else (1*i)
            
            # rectified data
            axes[nRowID, nColID].plot(arrData+nDataShift,
                                      color=lsRGB[i], lw=3, 
                                      alpha=dAlpha)
            # approximating shapes                                      
            if(bSourceEncoding is True and bPlotShape is True):                          
                axes[nRowID, nColID].plot(lsSourceShape[i]+nShapeShift,
                                          color='k',
                                          lw=1, alpha=dAlpha)
                                      
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else strSyncTitle)  +"(rect)", fontname=strFontName,
                fontsize=nFontSize)
                

                 
            if (tpYLim_rectified is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_rectified[0],
                                              tpYLim_rectified[1])
                
            # plot auxiliary line
            if(bPlotAuxiliaryLine is True):
                #coding wnd line
                for ln in xrange(0, len(arrData), nSCWndSize):
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
                    
            # decorate the axis  
            xTickLabels = axes[nRowID, nColID].xaxis.get_ticklabels()
            plt.setp(xTickLabels, fontname=strFontName,
                     size=nFontSize)
            yTickLabels = axes[nRowID, nColID].yaxis.get_ticklabels()
            plt.setp(yTickLabels, fontname=strFontName,
                     size=nFontSize)
        nCurrentRow += 1

    fig.suptitle(strFileName, fontname=strFontName, fontsize=nFontSize)
    plt.tight_layout()
    plt.show()
    
def sourceEncode(lsData_rectified, nSCWndSize, nNeighborWnd=3):
    """
        encoding data
        
        Parameters:
        ----------
        lsData_rectified:
            list of rectified EMG data
        nSCWndSize:
            source coding window size (in points)
        nNeighborWnd:
            searching area of nearby neighbors
            
        Returns:
        -------
        lsSourceCode:
            list of source code lists
        lsSourceCode_bin:
            list of source codes in binary
        lsSourceShape:
            list of source approximated shape arrays
        
    """
    
    lsSourceCode = []
    lsSourceCode_bin = []
    lsSourceShape = []
    for i, arrData in enumerate(lsData_rectified):
        lsDataSrcCode, arrDataShape = encoder.shapeEncoding(arrData,
                                                 nSCWndSize, nNeighborWnd)
        arrDataSrcCode = np.array(lsDataSrcCode)
        lsSourceCode.append(arrDataSrcCode)

        arrDataSrcCode_bin = ct.toBinaryArray(arrDataSrcCode, 2)
        lsSourceCode_bin.append(arrDataSrcCode_bin)
        
        lsSourceShape.append(arrDataShape)
        
        
    return lsSourceCode, lsSourceCode_bin, lsSourceShape


def interleaving(lsSourceCode_bin, nInterleaving):
    """
        interleave each data
        
        Note:
        -----
        To prevent the modification on original data, this function
        makes a copy of data.
    """
    lsSrcCode_bin_interlv = []
    for arrData in lsSourceCode_bin:
        arrData_interleaved = np.copy(arrData)
        arrData_interleaved, nPd = ct.interleave(arrData_interleaved,
                                             nInterleaving)
        lsSrcCode_bin_interlv.append(arrData_interleaved)
    return lsSrcCode_bin_interlv
   
def reconciliation(lsSrcCode_bin,
                   eng,
                   strCoder, n, k, m):
    """
        Perform reconciliation
    """
    dcReconciliation = {}
 
    # compute delta                         
    arrData_bin_user = lsSrcCode_bin[DATA_ID_USER]
    nPadding = n*m if strCoder==ecc.CODER_RS else n
    arrData_bin_user, nPd = ct.zeroPadding(arrData_bin_user, nPadding)
    arrDelta = ecc.computeDelta(eng, arrData_bin_user, 
                                n, k, m, strCoder)
                                
    # reconciliation btw user & payend
    arrData_bin_payend = lsSrcCode_bin[DATA_ID_PAYEND]
    arrData_bin_payend, nPd = ct.zeroPadding(arrData_bin_payend, nPadding)
    
    arrDeduced_bin_user_payend = ecc.reconciliate(eng, arrDelta,
                                      arrData_bin_payend,
                                      n, k, m, strCoder)
        
    dcReconciliation[DATA_ID_USER] = {DATA_1: arrData_bin_user, 
                           DATA_2: arrData_bin_payend,
                           DEDUCED_D1: arrDeduced_bin_user_payend}
                           
    # reconciliation btw attacker & payend
    arrData_bin_attacker = lsSrcCode_bin[DATA_ID_ATTACKER]
    arrData_bin_attacker, nPd = ct.zeroPadding(arrData_bin_attacker, nPadding)
    
    arrDeduced_bin_user_attacker = ecc.reconciliate(eng, arrDelta,
                                      arrData_bin_attacker,
                                      n, k, m, strCoder)
        
    dcReconciliation[DATA_ID_ATTACKER] = {DATA_1: arrData_bin_user, 
                           DATA_2: arrData_bin_attacker,
                           DEDUCED_D1: arrDeduced_bin_user_attacker}
                           
    return dcReconciliation
    
def evaluateReconciliation(lsSrcCode_bin_key, dcReconciliation):
    """
        Evaluate the performance of reconciliation
    """
    dcPerformance = {}    
    for i in [DATA_ID_USER, DATA_ID_ATTACKER]:
        # before reconciliation
        arrSrcCode1 = lsSrcCode_bin_key[i]
        arrSrcCode2 = lsSrcCode_bin_key[DATA_ID_PAYEND]
        nErrorBits_src, dBER_src =ct.computeBER(arrSrcCode1,
                                                arrSrcCode2)
        # after reconciliation
        arrData_bin_1 = dcReconciliation[i][DATA_1]
        arrDeduced_1 = dcReconciliation[i][DEDUCED_D1]

        nErrorBit_ec, dBER_ec =ct.computeBER(arrData_bin_1,
                                             arrDeduced_1)
                                             
        if(i == DATA_ID_USER):
            dcPerformance[BER_USER_SRC] = dBER_src
            dcPerformance[BER_USER_EC] = dBER_ec
            dcPerformance[ERR_USER_SRC] = nErrorBits_src
            dcPerformance[ERR_USER_EC] = nErrorBit_ec
        elif (i == DATA_ID_ATTACKER):
            dcPerformance[BER_ATTACKER_SRC] = dBER_src
            dcPerformance[BER_ATTACKER_EC] = dBER_ec
            dcPerformance[ERR_ATTACKER_SRC] = nErrorBits_src
            dcPerformance[ERR_ATTACKER_EC] = nErrorBit_ec
        else:
            raise ValueError("Unknown data id: %d" % i)
            
    return dcPerformance
    
     
def evaluateSingleData(strWorkingDir, strFileName,
                       dRectDuration = 2.0, dSMDuration=2.0, dSCDuration=0.1,
                       eng=None, strCoder=ecc.CODER_RS, 
                       m=4, r=5, n=15, k=5, nInterleaving=20,
                       bSourceEncoding=True, bReconciliation=True,
                       bOutputaData = False, lsOutputData = None,
                       bPlot=False, hKeyOutput = None):
    """
        Given the values of parameters, divide the generated sequence into 
        keys, and evaluate the performance of EMG-KEY.
        
        Parameters:
        ----------
        strWorkingDir:
            data directory
        strFileName:
            data file name
        dRectDuration:
            the duration of EMG rectification
        dSMDuration:
            the duration of smoothing function
        dSCDuration:
            the wnd duration of shape coding
        eng:
            matlab engine instance
        strCoder:
            name of error correction code
        m:
            number of bits per symbol (only for RS coder)
        r:
            maximal error bits
        n:
            length of code word
        k:
            length of code
        bPlot:
            plot the EMG signal if it is true
        hKeyOutput:
            file handle for outputing keys
        
        Returns:
        ----------
        lsResult:
            a list of dictions that contains evaluation results
    """
    dSamplingFreq = 240.0
    lsColumnNames = ['ch0', 'ch1', 'ch2']
    lsColumns2Inspect = ['ch0','ch1', 'ch2']
    
    # load data 
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    # process data
    nRawStart, nRawEnd = 0, -1
    lsData_raw, lsData_fft, \
        lsData_filtered, lsData_rectified = preprocess(dfData,
                                                       lsColumns2Inspect,
                                                       nRawStart,
                                                       nRawEnd,
                                                       dSamplingFreq, 
                                                       dSMDuration, 
                                                       dRectDuration)
        
    nSCWndSize = int(dSCDuration*dSamplingFreq) # src coding wnd size
    lsSourceCode = None        # source code
    lsSourceCode_bin = None    # source code in binary
    lsSourceShape = None       # shape of source
    
    if(bSourceEncoding):
        # source encoding
        lsSourceCode, lsSourceCode_bin, \
            lsSourceShape = sourceEncode(lsData_rectified, nSCWndSize)
        
        # compute entropy (only need entropy of user data)
        dSymbolEntropy = computeSymbolEntropy(lsSourceCode[DATA_ID_USER])    
                                    
        # interleaving            
        lsSrcCode_bin_interlv = interleaving(lsSourceCode_bin,
                                             nInterleaving)
        
        # write user's interleaved key to file
        if(hKeyOutput is not None):
            hKeyOutput.write(''.join(str(i) for i in \
                lsSrcCode_bin_interlv[DATA_ID_USER]) + "\n" )   
                
        lsResult = []
        if(bReconciliation):
            nKeySize = 60
            for i, nKeyStart in enumerate(range(0, len(lsSourceCode_bin[0]),\
                                                nKeySize) ):
                nKeyEnd = nKeyStart+nKeySize
                if nKeyEnd > len(lsSourceCode_bin[0]):
                    break
                
                # use part of sequence as key
                lsSrcCode_bin_key = [arrData[nKeyStart:nKeyEnd] \
                                        for arrData in lsSourceCode_bin]
                                            
                # prepare result dict
                dcKeyResult = {}
                dcKeyResult[FILE_NAME] = strFileName + '_k%d' % i
                dcKeyResult[M] = m
                dcKeyResult[R] = r
                dcKeyResult[N] = n
                dcKeyResult[K] = k
                dcKeyResult[KN_RATIO] = k*1.0/n
                dcKeyResult[CODER] = strCoder
                dcKeyResult[WND_RECT] = dRectDuration
                dcKeyResult[WND_SM] = dSMDuration
                dcKeyResult[WND_SC] = dSCDuration    
                dcKeyResult[ENTROPY] = dSymbolEntropy
                                            
                print "-->%s_%d_%d" % (strFileName, nKeyStart, nKeyEnd)
                
                # reconciliation
                dcReconciliation = reconciliation(lsSrcCode_bin_key,
                                                  eng,
                                                  strCoder, n, k, m)
                                                  
                           
                # evaluate reconciliation
                dcKeyResult[CODE_LEN_SRC] = len(lsSrcCode_bin_key[0])
                dcKeyResult[CODE_LEN_EC] = len(\
                    dcReconciliation[DATA_ID_USER][DATA_1])
                dcPerformance = evaluateReconciliation(lsSrcCode_bin_key,
                                                       dcReconciliation)
                for key, val in dcPerformance.iteritems():
                    dcKeyResult[key] = val
                
                # append to result
                lsResult.append(dcKeyResult)
                                                   
                    
    # compute mutual info.
    dcData_stat = {}
    for i in xrange(len(lsData_filtered)):
        arrCode1 = lsData_filtered[i]
        arrCode2 = lsData_filtered[(i+1)%len(lsData_filtered)]
        dMI = dt.computeMI(arrCode1, arrCode2, 200)
        dcData_stat["%d-%d" % (i, (i+1)%len(lsData_filtered)) ] = dMI

                        
    # prepare output data    
    if(bOutputaData is True and lsOutputData is not None):
        dfOutput = pd.DataFrame(lsData_rectified).T
        lsOutputData.append(dfOutput)
        
    # plot
    if(bPlot is True):
        plotData(lsData_raw, lsData_fft, lsData_filtered, lsData_rectified,
                 lsColumns2Inspect, dSamplingFreq, 
                 bPlotRawData=False, bPlotFFT=False, bPlotFiltered=False, 
                 bPlotRectified=True, 
                 bSourceEncoding=True, nSCWndSize=nSCWndSize,
                 lsSourceShape=lsSourceShape, bAnatation=False)

        
    return lsResult
        
    
    
if __name__ == '__main__':
    if('eng' not in globals() ):
        print "start matlab engine..."
        eng = matlab.engine.start_matlab()
    else:
        print "matlab engine is already existed."
    
    # setup
    strWorkingDir = "../../data/evaluation/reconciliation/"
    strFileName = 'yl_d1_g1_c1_20160324_151412.txt'
    
    strCoder = ecc.CODER_RS
    m = 4
    n = 2**m-1
    k = 11
    r = int(math.floor((n-k)/2.0) )
    nInterleaving = 25
    print "Coder=%s, n=%d, k=%d, m=%d, r=%d, interlv=%d" % \
            (strCoder, n, k, m, r, nInterleaving)
    
    # evaluate
    lsOutput = []                     
    lsResult = evaluateSingleData(strWorkingDir, strFileName,
                      dRectDuration=1., dSMDuration=1., dSCDuration=0.15,
                      eng=eng, strCoder=strCoder, 
                      m=m, r=r, n=n, k=k, nInterleaving = nInterleaving,
                      bSourceEncoding=True, bReconciliation=True,
                      bOutputaData=True, lsOutputData=lsOutput,
                      bPlot=True)
    
       
    dfResult = pd.DataFrame(lsResult)
    print "Evaluation on single data is finished."
    
    