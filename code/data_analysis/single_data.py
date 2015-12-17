# -*- coding: utf-8 -*-
"""
This script provides basic functions for analyzing a single data,
aslo, in its main body, it perform measurement on single data

@author: jason
"""
import signal_filter as sf
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
from scipy import stats
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

lsRGB = ['r', 'g', 'b']
lsCMYK = ['c', 'm', 'y']

def computeEnvelope(arrData, nWindow, nMinPeriods=None):
    """compute upper and lower envelope for given data"""
    arrUpperEnvelope = pd.rolling_max(pd.Series(arrData), window=nWindow,
                                      min_periods=nMinPeriods, center=True)
    arrLowerEnvelope = pd.rolling_min(pd.Series(arrData), window=nWindow,
                                     min_periods=nMinPeriods, center=True)
    return arrUpperEnvelope, arrLowerEnvelope
    
def findReponseEndIndex(arrData, dSamplingFreq, nResponses, 
                        dResponseDuration,
                        dRestDuration,
                        nSearchStartIndex = 100,
                        nDiscoutinousDistance = 100):
    """
        This function uses the std of data's variation range to
        determine the end indeces of responses.
        
        Parameters:
        ----
        arrValue: 
            the data
        dSamplingRate: 
            sampling rate of data
        nResponses: 
            number of responses
        nSearchStartIndex: 
            the start index of response searching
        nDiscoutinousDistance: 
            number of points btw two humps
        
        Returns:
        ----
        lsResponseEndIndex: 
            a list of end index of responses
        arrBandwidthSTD: 
            the reference data to find responses
    """
    nWindowSize = 50
    arrUpperEnvelope, \
    arrLowerEnvelope = computeEnvelope(arrData, nWindowSize)
    arrBandWidth = arrUpperEnvelope - arrLowerEnvelope
    arrBandwidthSTD = pd.rolling_std(pd.Series(arrBandWidth), 
                                     nWindowSize, center=True)
    
    lsPeaks = []
    
    # compute the minimal valume of peak candidates
    dMinPeakValume = (pd.Series(arrBandwidthSTD)).describe()['75%']
                    
                    
    # select points whose value is larger than dMinPeakValume
    arrPeakCandidateIndex = np.where(arrBandwidthSTD >= dMinPeakValume)[0]
    
    # only need points indexed after nSearchStartIndex
    arrPeakCandidateIndex = \
        arrPeakCandidateIndex[arrPeakCandidateIndex>=nSearchStartIndex]
        
    # find peaks
    lsPeaks = [(nIndex, arrBandwidthSTD[nIndex]) \
               for nIndex in arrPeakCandidateIndex]
    
    # search for the ending peak for responses
    lsEndingPeaks = []
    nSearchEnd = len(arrData)
    nSearchStart = int(nSearchEnd - (dResponseDuration/3.0)*dSamplingFreq)
    for i in xrange(nResponses):
        lsCandidates = [peak for peak in lsPeaks if \
                              peak[0]>=nSearchStart and \
                              peak[0]<nSearchEnd]
        tpEndingPeak = max(lsCandidates, key = operator.itemgetter(1) )
        lsEndingPeaks.append(tpEndingPeak)
        nSearchEnd = int(tpEndingPeak[0] - \
                     (dResponseDuration+dRestDuration)*dSamplingFreq + 2*dSamplingFreq)
        nSearchStart = int(nSearchEnd - (dResponseDuration/4.0)*dSamplingFreq)
    
    lsResponseEndIndex = sorted([peak[0] for peak in lsEndingPeaks], reverse=True)
    
    return lsResponseEndIndex, arrBandwidthSTD
            
            
    
def splitData(arrData, dSamplingFreq, nResponses, 
              nSegmentsPerRespsonse = 5, 
              dVibrationDuration = 1.4, 
              dIntervalDuration = 0.0, 
              dRestDuration = 1.0):
    """
        Split data into responses and then find segments in 
        each response.
        
        Parameters:
        ----
        arrData: 
            the data
        dSamplingFreq: 
            sampling frequency of data
        nResponses: 
            number of responses in this data
        nSegmentPerResponse: 
            number of segement per response
        dVibrationDuration: 
            duration of each vibration in seconds
        dIntervalDuration: 
            static duration btw vibrations in seconds
        dRestDuration: 
            rest duration btw responses in seconds
        
        Returns:
        ----
        lsResponses: 
            list of segment lists, each of which represent an response
        arrResponseEndIndex: 
            the ending index of each responses
        arrBandwidthSTD: 
            the std of data variation range 
    """
    # find the end of response via the std of variation
    dResponseDuration = nSegmentsPerRespsonse*dVibrationDuration + \
                        (nSegmentsPerRespsonse-1)*dIntervalDuration
    arrResponseEndIndex, \
    arrBandwidthSTD = findReponseEndIndex(arrData,
                                          dSamplingFreq,
                                          nResponses,
                                          dResponseDuration,
                                          dRestDuration,
                                          int(2*dSamplingFreq),
                                          int(1*dSamplingFreq) )
                                       
    lsResponses = []
    for nRespEndIndex in arrResponseEndIndex:
        if ( (nRespEndIndex- \
              nSegmentsPerRespsonse*(dVibrationDuration+dIntervalDuration) \
              *dSamplingFreq ) \
            < 0.0 ): 
            print "Invalid end index of response", arrResponseEndIndex
#            raise ValueError("Invalid end index of response.")
            
        
        lsSegments = []
        nSegmentEnd = nRespEndIndex
        nSegmentStart = nRespEndIndex
        nCount = 0
        while(nCount < nSegmentsPerRespsonse):
            nSegmentStart = int(nSegmentEnd - dSamplingFreq*dVibrationDuration ) 
            lsSegments.append( (nSegmentStart, nSegmentEnd) )
            nCount += 1
            
            nSegmentEnd = int(nSegmentStart - dSamplingFreq*dIntervalDuration)
            
        lsSegments.reverse()
        lsResponses.append(lsSegments)
        
    return lsResponses, arrResponseEndIndex, arrBandwidthSTD
    


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

def computeModulus(dfXYZ):
    """
    Given a 3-column dataframe, computes the modulus for each row

    Parameters
    ----------
    dfXYS: 
        data frame containing X, Y, Z data

    Returns
    ----------
    array of modulus
    """
    return (np.sqrt(np.power(dfXYZ.iloc[:,0], 2.0) +
                   np.power(dfXYZ.iloc[:,1], 2.0) +
                   np.power(dfXYZ.iloc[:,2], 2.0) )).values

def computeGravity(dfXYZ, nStart=0, nEnd=5000):
    """
        Given a 3-column acc data in some coordinates, computes the projection
        of gravity on each axis via averaging over stable state.

        Parameters
        ----------
        dfXYZ: 
            3-column ACC data frame
        nStart: 
            the start point of stable state
        nEnd: 
            the end point of stable state

        Returns
        ----------
        projection of gravity on X, Y, Z axies, and gravity, respectively
    """
    dAvgGX = np.average(dfXYZ.iloc[nStart:nEnd, 0])
    dAvgGY = np.average(dfXYZ.iloc[nStart:nEnd, 1])
    dAvgGZ = np.average(dfXYZ.iloc[nStart:nEnd, 2])
    return dAvgGX, dAvgGY, dAvgGZ, math.sqrt(dAvgGX**2+dAvgGY**2+dAvgGZ**2)

def removeGravity(dfXYZ, nStart=0, nEnd=1000):
    """
        This function compute the gravity via stable states,
        the remove it from data
#
        Parameteres
        -----------
        dfXYZ: 
            3-column ACC data frame
        nStart: 
            start point of stable state
        nEnd: 
            end point of stable state

        Returns
        ----------
        a 3-column ACC data frame without gravity
    """
    # compute gravity
    dAvgGX, dAvgGY, dAvgGZ, dGravity = computeGravity(dfXYZ, nStart, nEnd)
    srGravity = pd.Series([dAvgGX, dAvgGY,dAvgGZ], index=dfXYZ.columns)
    dfXYZ_noG = dfXYZ - srGravity

    return dfXYZ_noG
    
    
def findHeelStrike(arrData):
    """
        Given the accelerometer reading, find the heel-strike event
        
        Parameters
        --------
        arrData:
            accelerometers data
            
        Returns
        --------
        arrIndex:
            index of heel-strike event
        arrAccMagnitude
            acc magnitude
    """

def normCrossCorrelation(arrData0, arrData1):
    arrCrossCorr = np.correlate(arrData0, arrData1, mode='full')
    dDenominator = np.sqrt(np.dot(arrData0, arrData0) * \
                           np.dot(arrData1, arrData1) )
                           
    arrNormCrossCorr = arrCrossCorr/dDenominator
    nMaxIndex = np.argmax(abs(arrNormCrossCorr) )
    dCorr = arrNormCrossCorr[nMaxIndex]
    nLag = nMaxIndex - len(arrData0)
    return dCorr, nLag, arrNormCrossCorr
    
    
    
if __name__ == '__main__':
    
    # ---- load data ----
    strWorkingDir = "../../data/feasibility/"
    strFileName = "attacker_2"
    
    dSamplingFreq = 235.0
    lsColumnNames = ['ch0', 'ch1', 'ch2']
    
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    # ---- plot setup----
    lsColumns2Inspect = ['ch0','ch1', 'ch2']
    
    # look-and-feel
    nBasicFontSize = 18
    strBasicFontName = "Times new Roman"
    
    # plot raw 
    bPlotRawData = False
    tpYLim_raw = None
    
    # plot fft on raw data
    bPlotFFT = False
    nDCEnd = 2
    tpYLim_fft = None
    
    # plot filtered data
    bPlotFiltered = True
    tpYLim_filtered = None
    
    # plot fft on filtered data
    bPlotFFTonFiltered = False
    tpYLim_fft_filtered = None
    
    # plot statistics of filtered data
    bPlotStat = True
    tpYLim_stat = None
    
    # plot statistics of filtered data
    bPlotCoding = True
    tpYLim_coding = None
    
    # plot synchronized view
    bPlotSyncView = True
    strSyncTitle = "".join([s+"_" for s in lsColumns2Inspect] )
    
    # bAnatation
    bAnatation = True
    
    # ---- process data ----
    # raw
    nRawStart, nRawEnd = 0, -1
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
        arrNormalizedPower = abs(arrFFT)/(nSamples_fft*1.0)
        lsData_fft.append(arrNormalizedPower)
        
    # filtered
    nLowCut, nHighCut = 5, 45
    dPowerInterference = 50.0
    nOrder = 9
    nFilterShift = 50
    
    lsData_filtered = []
    for arrData in lsData_raw:
        arrNoise = sf.notch_filter(arrData, dPowerInterference-1., 
                                   dPowerInterference+1, 
                                   dSamplingFreq, order=nOrder)
        arrFiltered = arrData - arrNoise
        lsData_filtered.append(arrFiltered)
        
    # fft on filtered
    nSamples_fft_filtered = len(lsData_filtered[0])
    dRes_fft_filtered = dSamplingFreq*1.0/nSamples_fft_filtered
    lsData_fft_filtered = []
    for arrFiltered in lsData_filtered:
        arrFFT = fftpack.fft(arrFiltered)
        arrNormalizedPower = abs(arrFFT)/(nSamples_fft_filtered*1.0)
        lsData_fft_filtered.append(arrNormalizedPower)
        
    # statistics of data
    nWndSize = dSamplingFreq*0.5
    lsData_stat = []
    for arrData in lsData_filtered:
        arrStat = pd.rolling_mean(arrData, window=nWndSize, 
                                  min_periods=1)
        lsData_stat.append(arrStat)
        
    # coding
    lsData_coding = []
    for arrData in lsData_stat:
        arrStat = pd.rolling_quantile(arrData, window=nWndSize,
                                      quantile=0.5,
                                      min_periods=1)
        lsData_coding.append(arrStat)
        
    # correlation
    if (bAnatation is True):
        lsCorr_raw = []
        for i in xrange(len(lsData_raw) ):
            dCorr, p = stats.pearsonr(lsData_raw[i],
                                      lsData_raw[(i+1)%len(lsData_filtered)])
            lsCorr_raw.append(dCorr)
        print "lsCorr_raw", lsCorr_raw
            
        lsCorr_filtered = []
        for i in xrange(len(lsData_filtered) ):
            dCorr, p = stats.pearsonr( \
                lsData_filtered[i], 
                lsData_filtered[(i+1)%len(lsData_filtered)])
            lsCorr_filtered.append(dCorr)
        print "lsCorr_filtered: ", lsCorr_filtered
                
        lsCorr_stat = []
        for i in xrange(len(lsData_stat) ):
            dCorr, p = stats.pearsonr( \
                lsData_stat[i], 
                lsData_stat[(i+1)%len(lsData_stat)])
            lsCorr_stat.append(dCorr)
        print "lsCorr_stat", lsCorr_stat
            
            
                
    # ---- plot ---- 
    nRows = np.sum([bPlotRawData, bPlotFFT, bPlotFiltered, 
                    bPlotFFTonFiltered, bPlotStat, bPlotCoding] )
    nCols= len(lsColumns2Inspect) if bPlotSyncView is False else 1
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    
    nCurrentRow = 0
    
    # plot raw data
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
                else  strSyncTitle +"(raw)" ) )
            if (tpYLim_raw is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_raw[0], tpYLim_raw[1])
            axes[nRowID, nColID].grid('on')
            
        nCurrentRow += 1
        
    # plot FFT on raw
    if (bPlotFFT is True):
        for i, arrNormalizedPower in enumerate(lsData_fft):
            arrFreqIndex = np.linspace(nDCEnd*dRes_fft, 
                                       dSamplingFreq/2.0, 
                                       nSamples_fft/2-nDCEnd)
                                       
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (10*i)
            axes[nRowID, nColID].plot(arrFreqIndex,
                arrNormalizedPower[nDCEnd:nSamples_fft/2]-nVerticalShift,
                color=lsRGB[i], alpha=dAlpha)
            axes[nRowID, nColID].set_xticks(range(0, 
                                            int(dSamplingFreq/2), 10) )
            axes[nRowID, nColID].set_xlabel( \
                (lsColumns2Inspect[i]  if bPlotSyncView is False \
                else strSyncTitle +"(fft)" ) )
            if (tpYLim_fft is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_fft[0], tpYLim_fft[1])
            axes[nRowID, nColID].grid('on')

        nCurrentRow += 1
            
    # plot filtered data
    if(bPlotFiltered is True):
        for i, arrFiltered in enumerate(lsData_filtered):
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (80*i)
            axes[nRowID, nColID].plot(\
                arrFiltered[nFilterShift:]-nVerticalShift,
                color=lsRGB[i], alpha=dAlpha)
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else strSyncTitle +"(filtered)" ) )
            if (tpYLim_filtered is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_filtered[0],
                                              tpYLim_filtered[1])
            axes[nRowID, nColID].grid('on')
            
            if(bAnatation is True):
                axes[nRowID, nColID].annotate(\
                    'corr%d%d = %.2f'% \
                    (i, (i+1)%len(lsColumns2Inspect), lsCorr_filtered[i]), 
                    xy= (.5+i*0.2, .1) if bPlotSyncView else (.7, .1), 
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    verticalalignment='center')
        nCurrentRow += 1
        
    # plot FFT on filtered data
    if (bPlotFFTonFiltered is True):
        for i, arrNormalizedPower in enumerate(lsData_fft_filtered):
            arrFreqIndex = np.linspace(nDCEnd*dRes_fft_filtered, 
                                       dSamplingFreq/2.0, 
                                       nSamples_fft_filtered/2-nDCEnd)
                                       
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (5*i)
            axes[nRowID, nColID].plot(arrFreqIndex,
                arrNormalizedPower[nDCEnd: nSamples_fft_filtered/2.0]-nVerticalShift,
                color=lsRGB[i], alpha=dAlpha)
            axes[nRowID, nColID].set_xticks(range(0, 
                                            int(dSamplingFreq/2), 10) )
            axes[nRowID, nColID].set_xlabel( \
                (lsColumns2Inspect[i]  if bPlotSyncView is False \
                else strSyncTitle +"(fft@filtered)") )
            if (tpYLim_fft_filtered is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_fft_filtered[0], 
                                              tpYLim_fft_filtered[1])
            axes[nRowID, nColID].grid('on')
        nCurrentRow += 1
    
    
    # plot stat data
    if(bPlotStat is True):
        for i, arrStat in enumerate(lsData_stat):
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (50*i)
            axes[nRowID, nColID].plot(arrStat-nVerticalShift, 
                                      color=lsRGB[i], 
                                      alpha=dAlpha)
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else strSyncTitle +"(stat)" ) )
            if (tpYLim_stat is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_stat[0], tpYLim_stat[1])
            axes[nRowID, nColID].grid('on')
            
            if(bAnatation is True):
                axes[nRowID, nColID].annotate(\
                    'corr%d%d = %.2f'% \
                    (i, (i+1)%len(lsColumns2Inspect), lsCorr_stat[i]), 
                    xy= (.5+i*0.2, .1) if bPlotSyncView else (.7, .1), 
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    verticalalignment='center')
        nCurrentRow += 1
        
        
    # plot coding data
    if(bPlotCoding is True):
        for i, arrCoding in enumerate(lsData_coding):
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            nVerticalShift = 0 if bPlotSyncView is False else (50*i)
            axes[nRowID, nColID].plot(arrCoding-nVerticalShift, 
                                      color=lsRGB[i], 
                                      alpha=dAlpha)
            axes[nRowID, nColID].set_xlabel(\
                (lsColumns2Inspect[i] if bPlotSyncView is False \
                else strSyncTitle +"(coding)" ) )
            if (tpYLim_coding is not None):
                axes[nRowID, nColID].set_ylim(tpYLim_coding[0],
                                              tpYLim_coding[1])
            axes[nRowID, nColID].grid('on')
        nCurrentRow += 1
    
    plt.tight_layout()
    plt.show()
        