"""
Read data from serial and plot in real-time manner
"""

import signal_filter as sf

import sys
import serial
from threading import Thread, Event, Lock
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Button
import scipy.fftpack as fftpack
import datetime as dt

SAMPLING_FREQ = 250
MAX_PLOT_BUFF_SIZE = SAMPLING_FREQ*2                        # rx buff

FILTER_HIGHT_CUT = 20
FILTER_LOW_CUT = 5
FILTER_ORDER = 9

g_dataQueue = deque(maxlen=MAX_PLOT_BUFF_SIZE)  # rx buffer
g_dataRxEvent = Event()                         # rx event
g_dataWrittingEvent = Event()                   # writting event
g_dataQueueLock = Lock()                        # lock for rx buffer
g_lsColors = ['r', 'b', 'g', 'c', 'm', 'k']

def checkDataFormat(arrValues):
    bValid = True
    try:
        for d in arrValues:
            float(d)
    except ValueError as exp:
        print exp
        bValid = False
    return bValid
    
def onBtRecord(event):
    '''
        handler when buttom is cliked
    '''
    if(g_dataWrittingEvent.is_set() ):
        g_dataWrittingEvent.clear()
        print("recording is stopped.")
        
    else:
        g_dataWrittingEvent.set()
        print("recording is started.")


def dataRx(ser, nChannels,
           dataQueue, rxEvent, dataLock, 
           writeEvent, strFilePath):
  '''
      handler for receiving data 
  '''
  # open serial
  try:
    hFile = None

    # read 
    while rxEvent.is_set():
      strLine = ser.readline()

      # save to file
      if(writeEvent.is_set() ):
          if (hFile is None):
              strFileName = dt.datetime.strftime(dt.datetime.now(), 
                                                 '%Y%m%d_%H%M%S')
              hFile = open(strFilePath+strFileName+".txt", 'w+')
          hFile.write(strLine)
      elif (hFile is not None):
          hFile.flush()
          hFile.close()
          hFile = None
          
      # add to plot buff
      values = strLine.split(",")
      if (len(values) >= nChannels and checkDataFormat(values) is True):
          dataLock.acquire()
          dataQueue.append(values)
          dataLock.release()
      
  finally:
    # close file
    if (hFile is not None):
        hFile.flush()
        hFile.close()


def onDraw(frameNum, dataQueue, nChannels,
           dSamplingFreq, 
           lsAx_raw,
           lsAx_fft,
           lsAx_filtered,
           lsAx_fft_filtered,
           bt, dataLock, writtingEvent):

    # get a copy of data
    lsData_raw = []
    dataLock.acquire()
    for i in range(nChannels):
        arrData = np.array([d[i+1] for d in dataQueue], dtype=np.float64)
        lsData_raw.append(arrData)
    dataLock.release()
    
    # update time-domain plot
    for i, ax in enumerate(lsAx_raw):
        arrData = lsData_raw[i]
        ax.set_data(range(len(arrData) ), arrData)

    # filter data
    for i, ax in enumerate(lsAx_filtered):
        arrData = lsData_raw[i]
        arrData_filtered = sf.butter_lowpass_filter(arrData, 
                                                    FILTER_HIGHT_CUT,
                                                    dSamplingFreq,
                                                    order=FILTER_ORDER)

        ax.set_data(range(len(arrData_filtered) ), arrData_filtered)

    # update spectral plot
    nSamples = len(lsData_raw[0])
    if (nSamples >= dSamplingFreq):
        dResolution = dSamplingFreq*1.0/nSamples
        nDCEnd = 5
        arrFreqIndex = np.linspace(nDCEnd*dResolution,
                                   dSamplingFreq/2.0, 
                                   nSamples/2-nDCEnd)
        for i, ax in enumerate(lsAx_fft):
          arrFFT = fftpack.fft(lsData_raw[i])
          arrNormalizedPower = np.abs(arrFFT)/nSamples*1.0
          ax.set_data(arrFreqIndex, arrNormalizedPower[nDCEnd:nSamples/2])
        
        for i, ax in enumerate(lsAx_fft_filtered):
            arrData = lsData_raw[i]
            arrData_filtered = sf.butter_lowpass_filter(arrData,
                                                        FILTER_HIGHT_CUT,
                                                        dSamplingFreq,
                                                        order=FILTER_ORDER)
            arrFFT = fftpack.fft(arrData_filtered)
            arrNormalizedPower = np.abs(arrFFT)/nSamples*1.0
            ax.set_data(arrFreqIndex, arrNormalizedPower[nDCEnd:nSamples/2])
        
    
    strLabel = "stop" if (writtingEvent.is_set()) else "record"
    bt.label.set_text(strLabel)


def main():
  if (len(sys.argv) != 3 ):
      print "Usage: script_name port channel_number"
      return
      
  dataRxThread = None
  ser = None
  try:
    # setup 
    nPort = int(sys.argv[1])
    nChannel = int(sys.argv[2])
    nBaudRate = 57600
    strDataPath = "../../data/"
    
    # open serial port
    print("Try to open COM %d..." % nPort)
    ser = serial.Serial(nPort, nBaudRate)
    print("Serial %s is open." % ser.name)
    
    # create & start rx thread
    g_dataRxEvent.set()
    dataRxThread = Thread(target=dataRx, args=(ser, nChannel,
                          g_dataQueue,
                          g_dataRxEvent, g_dataQueueLock, 
                          g_dataWrittingEvent, strDataPath) )
    dataRxThread.start()

    # set up plot
    fig, axes = plt.subplots(nrows=4, ncols=nChannel, squeeze=True)

    # setup look-and-feel
    for nCol in xrange(nChannel):
      axes[0, nCol].set_xlim(0, MAX_PLOT_BUFF_SIZE)
      axes[0, nCol].set_ylim(-100, 1500)
      axes[0, nCol].set_xlabel('raw data')

      axes[1, nCol].set_xlim(0, SAMPLING_FREQ/2.0+5)
      axes[1, nCol].set_ylim(0, 30)
      axes[1, nCol].set_xticks(range(0, SAMPLING_FREQ/2, 5) )
      axes[1, nCol].set_xlabel('FFT')

      axes[2, nCol].set_xlim(0, MAX_PLOT_BUFF_SIZE)
      axes[2, nCol].set_ylim(-100, 1500)
      axes[2, nCol].set_xlabel('filtered data')

      axes[3, nCol].set_xlim(0, SAMPLING_FREQ/2.0+5)
      axes[3, nCol].set_ylim(0, 30)
      axes[3, nCol].set_xticks(range(0, SAMPLING_FREQ/2, 5) )
      axes[3, nCol].set_xlabel('FFT on filtered')

    # create buttom
    plt.subplots_adjust(bottom=0.2)
    axBt = plt.axes([0.7, 0.05, 0.1, 0.075])
    btRecord = Button(axBt, 'record')
    btRecord.on_clicked(onBtRecord)
             
    # set axes for each plot
    lsAx_raw = []
    for i in xrange(nChannel):
        ax, = axes[0, i].plot([], [], color=g_lsColors[i])
        lsAx_raw.append(ax)
        
    lsAx_fft = []
    for i in xrange(nChannel):
        ax, = axes[1, i].plot([], [], color=g_lsColors[i])
        lsAx_fft.append(ax)

    lsAx_filtered = []
    for i in xrange(nChannel):
        ax, = axes[2, i].plot([], [], color=g_lsColors[i])
        lsAx_filtered.append(ax)
    
    lsAx_fft_filtered = []
    for i in xrange(nChannel):
        ax, = axes[3, i].plot([], [], color=g_lsColors[i])
        lsAx_fft_filtered.append(ax)


    # create animation
    anim = animation.FuncAnimation(fig, onDraw, 
                                   fargs=(g_dataQueue, nChannel,
                                          SAMPLING_FREQ,
                                          lsAx_raw,
                                          lsAx_fft,
                                          lsAx_filtered,
                                          lsAx_fft_filtered,
                                          btRecord, 
                                          g_dataQueueLock,
                                          g_dataWrittingEvent), 
                                   interval=100)
    

    # show plot
    plt.show()

  finally:
    # end rx thread
    g_dataRxEvent.clear()

    if (dataRxThread is not None):
      dataRxThread.join()
    
    # close serial port
    if (ser is not None):
        ser.close()
        print("Serial is closed")


# call main
if __name__ == '__main__':
    main()
