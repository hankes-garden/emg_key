# -*- coding: utf-8 -*-
"""
This script provides basic functions for analyzing a single data.

@author: jason
"""
import source_encoding as encoder


import signal_filter as sf
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats
from collections import deque
import itertools

lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']


RECTIFY_ARV = 'ARV'
RECTIFY_RMS = 'RMS'

MEAN_FREQ = "mean_freq"
MEDIAN_FREQ = "median_freq"


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

def computeEnvelope(arrData, nWindow, nMinPeriods=None):
    """compute upper and lower envelope for given data"""
    arrUpperEnvelope = pd.rolling_max(pd.Series(arrData), window=nWindow,
                                      min_periods=nMinPeriods, center=True)
    arrLowerEnvelope = pd.rolling_min(pd.Series(arrData), window=nWindow,
                                     min_periods=nMinPeriods, center=True)
    return arrUpperEnvelope, arrLowerEnvelope
    
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
            
            
def loadData(strWorkingDir, strFileName, lsColumnNames, strFileExt = '.txt'):
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
    

def normalizedCrossCorr(arrData0, arrData1):
    """
        Compute the normalized cross correlation
    """
    arrCrossCorr = np.correlate(arrData0, arrData1, mode='full')
    dDenominator = np.sqrt(np.dot(arrData0, arrData0) * \
                           np.dot(arrData1, arrData1) )
                           
    arrNormCrossCorr = arrCrossCorr/dDenominator
    nMaxIndex = np.argmax(abs(arrNormCrossCorr) )
    dCorr = arrNormCrossCorr[nMaxIndex]
    nLag = nMaxIndex - len(arrData0)
    return dCorr, nLag, arrNormCrossCorr
        
def slidingCorrelation(arrData1, arrData2, nWndSize):
    """
        compute correlation btw two signals with sliding window
    """
    nLen = min(len(arrData1), len(arrData2) )
    
    dTotalCorr = 0.0
    for nStart in xrange(0, nLen-nWndSize, nWndSize):
        nEnd = min(nStart + nWndSize, nLen)
        dCorr, p = stats.pearsonr(arrData1[nStart: nEnd], 
                                  arrData2[nStart: nEnd])
        dTotalCorr += dCorr
    dTotalCorr = dTotalCorr * nWndSize / nLen
    return dTotalCorr
     
     
if __name__ == '__main__':
    """
        examine single data
    """
    # ---- load data ----
    strWorkingDir = "../../data/feasibility/"
    strFileName = "ww_20160120_220019"
    
    dSamplingFreq = 236.0
    lsColumnNames = ['ch0', 'ch1', 'ch2']
    
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    # ---- setup----
    lsColumns2Inspect = ['ch0','ch1', 'ch2']
    
#==============================================================================
# process data
#==============================================================================
    # ---- raw ----
    nRawStart, nRawEnd = 0, -1
    lsData_raw = []
    for col in lsColumns2Inspect:
        arrData = dfData[col].iloc[nRawStart: nRawEnd]
        lsData_raw.append(arrData)
        
    # ---- fft ----
    nDCEnd = 2
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
        arrNoise = sf.notch_filter(arrData, dPowerInterference-2., 
                                   dPowerInterference+2., 
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
    dRectDuration = 0.5
    lsData_rectified = []
    for arrData in lsData_filtered:
        arrRect = rectifyEMG(arrData, dSamplingFreq, 
                             dWndDuration=dRectDuration,
                             method=RECTIFY_ARV)
        lsData_rectified.append(arrRect)
        
    # statistics of data
    nCodingWndSize = int(0.5*dSamplingFreq)
    dcData_stat = {}
    lsSegments = dcSegments[strFileName]
    for i in xrange(len(lsData_rectified)):
        arrData1 = lsData_rectified[i]
        arrData2 = lsData_rectified[(i+1)%len(lsData_rectified)]
        
        dTotalCorr = 0.0
        for nStart, nEnd in lsSegments[:]:
            arrSeg1 = arrData1[nStart: nEnd]
            arrSeg2 = arrData2[nStart: nEnd]
            dCorr = slidingCorrelation(arrSeg1, arrSeg2,
                                       nWndSize=nCodingWndSize)
            dTotalCorr += abs(dCorr)
        dTotalCorr = dTotalCorr / len(lsSegments[:])
        dcData_stat["%d-%d" % (i, (i+1)%len(lsData_rectified)) ] = dTotalCorr
        
        
    print "correlation: \n", dcData_stat, "\n"
        
    # coding
    bCoding = False
    if(bCoding):
        lsData_coding = []
        for arrData in lsData_rectified:
            # segment
            lsSegments = dcSegments[strFileName]
            
            # coding
            lsSegmentCode = []
            for nStart, nEnd in lsSegments:
                arrEMGSeg = arrData[nStart:nEnd]
                lsCode = encoder.shapeEncoding(arrEMGSeg, nCodingWndSize)
                lsSegmentCode += lsCode
            lsData_coding.append(np.array(lsSegmentCode) )
            
        # coding performance
        for i in xrange(len(lsData_coding) ):
            arrCode1 = lsData_coding[i]
            arrCode2 = lsData_coding[(i+1) % len(lsData_coding) ]
            dBER = encoder.computeBER(arrCode1, arrCode2)
            print("BER(%d, %d)=%.2f" % \
                    (i, (i+1) % len(lsData_coding), dBER) )
                
#==============================================================================
# plot
#==============================================================================
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
            nVerticalShift = 0 if bPlotSyncView is False else (400*i)
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
    if (bPlotFFT is True):
        for i, arrPSD in enumerate(lsData_fft):
            arrFreqIndex = np.linspace(nDCEnd*dRes_fft, 
                                       dSamplingFreq/2.0, 
                                       nSamples_fft/2-nDCEnd)
                                       
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (10*i)
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
            nVerticalShift = 0 if bPlotSyncView is False else (3*i)
            axes[nRowID, nColID].plot(arrFreqIndex,
                arrPSD[nDCEnd: int(nSamples_fft_filtered/2.0)] \
                    -nVerticalShift,
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
            nVerticalShift = 0 if bPlotSyncView is False else (2*i)
            axes[nRowID, nColID].plot(arrData+nVerticalShift,
                                      color=lsRGB[i], 
                                      alpha=dAlpha)
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else strSyncTitle)  +"(rect)" )
            if (tpYLim_rectified is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_rectified[0],
                                              tpYLim_rectified[1])
            axes[nRowID, nColID].grid('on')
                
            # plot auxiliary line
            if(bPlotAuxiliaryLine is True):
                lsSegmentIndex = dcSegments[strFileName]
                for nStart, nEnd in lsSegmentIndex:
                    # segment line
                    axes[nRowID, nColID].axvline(nStart, color='k', 
                                                 ls='--', lw=2)
                    axes[nRowID, nColID].axvline(nEnd, color='k', 
                                                 ls='--', lw=2)
                    
                    #coding wnd line
                    for ln in xrange(nStart, nEnd, nCodingWndSize):
                        axes[nRowID, nColID].axvline(ln, color='c', ls='--')
            
            # anatation
            if(bAnatation is True):
                strKey = "%d-%d" % (i, (i+1)%len(lsData_rectified) )
                axes[nRowID, nColID].annotate(\
                    'corr_%s = %.2f'% (strKey, dcData_stat[strKey]), 
                    xy= (.5+i*0.2, .85) if bPlotSyncView \
                        else tpAnatationXYCoor, 
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    verticalalignment='center')
        nCurrentRow += 1
        

    fig.suptitle(strFileName, fontname=strFontName, fontsize=nFontSize)
    plt.tight_layout()
    plt.show()
    
