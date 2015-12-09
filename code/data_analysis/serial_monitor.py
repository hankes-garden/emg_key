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

g_dataQueue = deque(maxlen=MAX_PLOT_BUFF_SIZE)  # rx buffer
g_dataRxEvent = Event()                         # rx event
g_dataWrittingEvent = Event()                   # writting event
g_dataQueueLock = Lock()                        # lock for rx buffer


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


def dataRx(ser, 
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
      dataLock.acquire()
      dataQueue.append(values)
      dataLock.release()
      
  finally:
    # close file
    if (hFile is not None):
        hFile.flush()
        hFile.close()


def onDraw(frameNum, dataQueue, dSamplingFreq, 
           ax_t0, ax_t1, 
           ax_f0, ax_f1,
           ax_t0_filtered, ax_t1_filtered,
           ax_f0_filtered, ax_f1_filtered,
           bt, dataLock, writtingEvent):

    # get a copy of data
    dataLock.acquire()
    arrCh0 = np.array([d[1] for d in dataQueue], dtype=np.float64)
    arrCh1 = np.array([d[2] for d in dataQueue], dtype=np.float64)
    dataLock.release()

    # update time-domain plot
    nSamples = len(arrCh0)
    ax_t0.set_data(range(nSamples), arrCh0)
    ax_t1.set_data(range(nSamples), arrCh1)

    # filter data
    arrCh0_filtered = sf.butter_lowpass_filter(arrCh0, 40,
                                               dSamplingFreq, order=20)
    arrCh1_filtered = sf.butter_lowpass_filter(arrCh1, 40, 
                                               dSamplingFreq, order=20)

    ax_t0_filtered.set_data(range(nSamples), arrCh0_filtered)
    ax_t1_filtered.set_data(range(nSamples), arrCh1_filtered)

    # update spectral plot
    if (nSamples >= dSamplingFreq):
      dResolution = dSamplingFreq*1.0/nSamples
      nDCEnd = 5
      arrFreqIndex = np.linspace(nDCEnd*dResolution,
                                 dSamplingFreq/2.0, 
                                 nSamples/2-nDCEnd)

      arrCh0_f = abs(fftpack.fft(arrCh0) )/ (nSamples*1.0)
      arrCh1_f = abs(fftpack.fft(arrCh1) )/ (nSamples*1.0)
      ax_f0.set_data(arrFreqIndex, arrCh0_f[nDCEnd:nSamples/2])
      ax_f1.set_data(arrFreqIndex, arrCh1_f[nDCEnd:nSamples/2])

      arrCh0_f_t = abs(fftpack.fft(arrCh0_filtered) )/ (nSamples*1.0)
      arrCh1_f_t = abs(fftpack.fft(arrCh1_filtered) )/ (nSamples*1.0)
      ax_f0_filtered.set_data(arrFreqIndex, arrCh0_f_t[nDCEnd:nSamples/2])
      ax_f1_filtered.set_data(arrFreqIndex, arrCh1_f_t[nDCEnd:nSamples/2])

      

    
    strLabel = "stop" if (writtingEvent.is_set()) else "record"
    bt.label.set_text(strLabel)


def main():
  dataRxThread = None
  ser = None
  try:
    # setup 
    nPort = int(sys.argv[1])
    nBaudRate = 57600
    strDataPath = "../../data/"
    
    # open serial port
    print("Try to open COM %d..." % nPort)
    ser = serial.Serial(nPort, nBaudRate)
    print("Serial %s is open." % ser.name)
    
    # create & start rx thread
    g_dataRxEvent.set()
    dataRxThread = Thread(target=dataRx, args=(ser, g_dataQueue,
                          g_dataRxEvent, g_dataQueueLock, 
                          g_dataWrittingEvent, strDataPath) )
    dataRxThread.start()

    # set up plot
    fig, axes = plt.subplots(nrows=4, ncols=2, squeeze=True)

    # setup look-and-feel
    for nCol in range(2):
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


    plt.subplots_adjust(bottom=0.2)

    # create buttom                  
    axBt = plt.axes([0.7, 0.05, 0.1, 0.075])
    btRecord = Button(axBt, 'record')
    btRecord.on_clicked(onBtRecord)
             
    # set axes for each plot
    ax_t0, = axes[0, 0].plot([], [], color='r')
    ax_t1, = axes[0, 1].plot([], [], color = 'b')

    ax_f0, = axes[1, 0].plot([], [], color='r')
    ax_f1, = axes[1, 1].plot([], [], color='b')

    ax_t0_filtered, = axes[2, 0].plot([], [], color='r')
    ax_t1_filtered, = axes[2, 1].plot([], [], color = 'b')

    ax_f0_filtered, = axes[3, 0].plot([], [], color='r')
    ax_f1_filtered, = axes[3, 1].plot([], [], color='b')

    


    # create animation
    anim = animation.FuncAnimation(fig, onDraw, 
                                   fargs=(g_dataQueue, SAMPLING_FREQ,
                                          ax_t0, ax_t1, 
                                          ax_f0, ax_f1, 
                                          ax_t0_filtered, ax_t1_filtered,
                                          ax_f0_filtered, ax_f1_filtered,
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
