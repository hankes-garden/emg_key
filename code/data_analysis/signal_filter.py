# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, lfilter, freqz, filter_design, filtfilt
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def notch_filter(data, lowcut, highcut, fs, order=5, rs=10):
    # Chebyshev notch filter centered on 50Hz
    nyquist = fs / 2.0
    b, a = filter_design.iirfilter(order, 
                                   (lowcut/nyquist, highcut/nyquist),
                                   rs=rs,
                                   ftype='cheby2')

    # filter the signal
    arrFiltered = filtfilt(b, a, data)
    
    return arrFiltered


if __name__ == "__main__":

    # plt rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Filter a noisy signal.
    T = 0.05
    res = T * fs
    t = np.linspace(0, T, res, endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 100 * t)
    x += 0.05 * np.cos(2 * np.pi * 600 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='raw signal')

    y = butter_lowpass_filter(x, 300, fs, order=6)
    plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
    
    # plt the frequency response for a few different orders.
    plt.figure()
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    
