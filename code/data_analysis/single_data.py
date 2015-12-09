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
    arrCrossCorr = sig.correlate(arrData0, arrData1, mode='full')
    dDenominator = np.sqrt(np.dot(arrData0, arrData0) * \
                           np.dot(arrData1, arrData1) ) 
    return arrCrossCorr/dDenominator
    
    
    
if __name__ == '__main__':
    
    # data 
    dSamplingFreq = 250.0
    lsColumnNames = ['ch0', 'ch1']
    
    strWorkingDir = "../../data/feasibility/"
    strFileName = "qy_1"
    
    
    dfData = loadData(strWorkingDir, strFileName, lsColumnNames)
    
    # plot setup
    nBasicFontSize = 16
    strBasicFontName = "Times new Roman"
    lsAxis2Inspect = ['ch0',]
    lsColors = lsRGB*int(math.ceil(len(lsAxis2Inspect)/3.0) )
    
    # raw data
    bPlotRawData = True
    nRawStart, nRawEnd = 0, -1
    
    # fft on raw data
    bPlotFFT = True
    nFFTStart = nRawStart
    nFFTEnd = nRawEnd
    nDCEnd = 2
    tpFFTYLim = (0, 5)
    
    # filtered data
    bPlotFiltered = True
    nLowCut = 5
    nHighCut = 20
    nOrder = 9
    nShift = 50
    
    # fft on filtered data
    bPlotFFTonFiltered = True
    tpFFTFilteredYLim = (0, 5)
    
    # statistics of filtered data
    bPlotStat = False
    
    
    # plot synchronized view
    bPlotSyncView = False
    
    # bAnatation
    bAnatation = True
    
    
    # create fig 
    nRows = np.sum([bPlotRawData, bPlotFFT, bPlotFiltered, 
                    bPlotFFTonFiltered, bPlotStat] )
    nCols= len(lsAxis2Inspect) if bPlotSyncView is False else 1
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, squeeze=False)
    nCurrentRow = 0
    
    # plot raw data
    arrComp0, arrComp1 = None, None
    if(bPlotRawData is True):
        for i, col in enumerate(lsAxis2Inspect):
            arrData = dfData[col].iloc[nRawStart:nRawEnd]
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            axes[nRowID, nColID].plot(arrData, color=lsColors[i], 
                                      alpha=dAlpha)
            axes[nRowID, nColID].set_xlabel( (col if bPlotSyncView is False \
                else ("".join(lsAxis2Inspect) ) ) +"(raw)" )
            axes[nRowID, nColID].set_ylim(0, 1000)
            axes[nRowID, nColID].grid('on')
            
            if (i == 0):
                arrComp0 = arrData
            elif (i == 1):
                arrComp1 = arrData
            else:
                pass
            
        if(bPlotSyncView):
            arrNormCrossCorr = normCrossCorrelation(arrComp0, arrComp1)
            nMaxIndex = np.argmax(arrNormCrossCorr)
            dCorr = arrNormCrossCorr[nMaxIndex]
            nLag = nMaxIndex - len(arrComp0)
            axes[nRowID, nColID].annotate("correlation=%.2f, lag=%d"%\
                                          (dCorr, nLag),
                                          xy=(0.6,1.0),
                                          textcoords='axes fraction')
        nCurrentRow += 1
        
    # plot FFT on raw
    if (bPlotFFT is True):
        arrComp0, arrComp1 = None, None
        for i, col in enumerate(lsAxis2Inspect):
            arrData = dfData[col].iloc[nFFTStart:nFFTEnd]
            nSamples = len(arrData)
            arrFFT = fftpack.fft(arrData)
            arrNormalizedPower = abs(arrFFT)/(nSamples*1.0)
            arrNorPower_sm = pd.rolling_mean(arrNormalizedPower, 5, 1)
            dResolution = dSamplingFreq*1.0/nSamples
            arrFreqIndex = np.linspace(nDCEnd*dResolution, 
                                       dSamplingFreq/2.0, 
                                       nSamples/2-nDCEnd)
                                       
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            axes[nRowID, nColID].plot(arrFreqIndex,
                                      arrNormalizedPower[nDCEnd:nSamples/2],
                                      color=lsColors[i],
                                      alpha=dAlpha)
            axes[nRowID, nColID].set_xticks(range(0, 
                                            int(dSamplingFreq/2), 10) )
            axes[nRowID, nColID].set_xlabel( (col if bPlotSyncView is False \
                else ("".join(lsAxis2Inspect) ) ) +"(fft)" )
            axes[nRowID, nColID].set_ylim(tpFFTYLim)
            
            if (i == 0):
                arrComp0 =  arrNormalizedPower[nDCEnd:nSamples/2]
            elif (i == 1):
                arrComp1 =  arrNormalizedPower[nDCEnd:nSamples/2]
            else:
                pass
            
        if(bPlotSyncView):
            dCorr, dPval = stats.pearsonr(arrComp0, arrComp1)
            axes[nRowID, nColID].annotate("correlation=%.2f"%dCorr,
                                          xy=(0.6,1.0),
                                          textcoords='axes fraction')
        nCurrentRow += 1
            
    # plot filtered data
    if (bPlotFiltered is True):
        for i, col in enumerate(lsAxis2Inspect):
            arrData = dfData[col].iloc[nRawStart:nRawEnd]
            arrFiltered= sf.butter_lowpass_filter(arrData, nHighCut,
                                                   dSamplingFreq,
                                                   order=nOrder)
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            axes[nRowID, nColID].plot(arrFiltered[nShift:], color=lsColors[i],
                                      alpha = dAlpha)
            axes[nRowID, nColID].set_xlabel( (col if bPlotSyncView is False \
                else ("".join(lsAxis2Inspect) ) ) +"(filtered)" )
#            axes[nRowID, nColID].set_ylim(300, 800)
            if (i == 0):
                arrComp0 = arrFiltered[100:]
            elif (i == 1):
                arrComp1 = arrFiltered[100:]
            else:
                pass
            
        if(bPlotSyncView):
            arrNormCrossCorr = normCrossCorrelation(arrComp0, arrComp1)
            nMaxIndex = np.argmax(arrNormCrossCorr)
            dCorr = arrNormCrossCorr[nMaxIndex]
            nLag = nMaxIndex - len(arrComp0)
            axes[nRowID, nColID].annotate("correlation=%.2f, lag=%d"%\
                                          (dCorr, nLag),
                                          xy=(0.6,1.0),
                                          textcoords='axes fraction')
        nCurrentRow += 1
        
    # plot FFT on filtered data
    if (bPlotFFTonFiltered is True):
        for i, col in enumerate(lsAxis2Inspect):
            arrData = dfData[col].iloc[nFFTStart:nFFTEnd]
            arrFiltered= sf.butter_lowpass_filter(arrData, nHighCut,
                                                   dSamplingFreq,
                                                   order=nOrder)
            nSamples = len(arrFiltered)
            arrFFT = fftpack.fft(arrFiltered[nShift:])
            arrNormalizedPower = abs(arrFFT)/(nSamples*1.0)
            arrNorPower_sm = pd.rolling_mean(arrNormalizedPower, 5, 1)
            dResolution = dSamplingFreq*1.0/nSamples
            arrFreqIndex = np.linspace(nDCEnd*dResolution, 
                                       dSamplingFreq/2.0, 
                                       nSamples/2-nDCEnd)
                                       
            nRowID = nCurrentRow
            nColID = i if bPlotSyncView is False else 0
            dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
            axes[nRowID, nColID].plot(arrFreqIndex,
                                      arrNormalizedPower[nDCEnd:nSamples/2],
                                      color=lsColors[i], alpha=dAlpha)
            axes[nRowID, nColID].set_xticks(range(0, 
                                            int(dSamplingFreq/2), 10) )
            axes[nRowID, nColID].set_xlabel( (col if bPlotSyncView is False \
                else ("".join(lsAxis2Inspect) ) ) +"(fft@filtered)" )
            axes[nRowID, nColID].set_xlim(0, nHighCut)
            axes[nRowID, nColID].set_ylim(tpFFTFilteredYLim)

            if (i == 0):
                arrComp0 = arrNormalizedPower[nDCEnd:nSamples/2]
            elif (i == 1):
                arrComp1 = arrNormalizedPower[nDCEnd:nSamples/2]
            else:
                pass
        nCurrentRow += 1
    
    # plot statistics of filtered data
    if (bPlotStat is True):
        for i, col in enumerate(lsAxis2Inspect):
                arrData = dfData[col].iloc[nRawStart:nRawEnd]
                arrFiltered= sf.butter_lowpass_filter(arrData, nHighCut,
                                                       dSamplingFreq,
                                                       order=nOrder)[nShift:]
                arrStat = pd.rolling_mean(arrFiltered, window=100,
                                          min_periods=1)
                nRowID = nCurrentRow
                nColID = i if bPlotSyncView is False else 0
                dAlpha = 1.0 if bPlotSyncView is False else (1.0-i*0.3)
                                                           
                axes[nRowID, nColID].plot(arrStat,
                                          color=lsColors[i],
                                          alpha=dAlpha)
                axes[nRowID, nColID].set_xlabel( (col \
                    if bPlotSyncView is False \
                    else ("".join(lsAxis2Inspect) ) ) +"(stats)" )
    #            axes[nRowID, nColID].set_ylim(300, 800)
                if (i == 0):
                    arrComp0 = arrFiltered[nShift:]
                elif (i == 1):
                    arrComp1 = arrFiltered[nShift:]
                else:
                    pass
            
        if(bPlotSyncView):
            arrNormCrossCorr = normCrossCorrelation(arrComp0, arrComp1)
            nMaxIndex = np.argmax(arrNormCrossCorr)
            dCorr = arrNormCrossCorr[nMaxIndex]
            nLag = nMaxIndex - len(arrComp0)
            axes[nRowID, nColID].annotate("correlation=%.2f, lag=%d"%\
                                          (dCorr, nLag),
                                          xy=(0.6,1.0),
                                          textcoords='axes fraction')
        nCurrentRow += 1
                                                 
    
    
    # set fig look        
    fig.suptitle(strFileName, fontname=strBasicFontName,
                 fontsize=nBasicFontSize)
    plt.tight_layout()
    plt.show()
        