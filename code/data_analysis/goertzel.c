# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport cos, M_PI

cpdef double goertzel(double[:] x, double ft, double fs=1.):
    """
    The Goertzel algorithm is an efficient method for evaluating single terms
    in the Discrete Fourier Transform (DFT) of a signal. It is particularly
    useful for measuring the power of individual tones.

    Arguments
    ----------
        x   double array [nt,]; the signal to be decomposed
        ft  double scalar; the target frequency at which to evaluate the DFT
        fs  double scalar; the sample rate of x (same units as ft, default=1)

    Returns
    ----------
        p   double scalar; the DFT coefficient corresponding to ft

    See: <http://en.wikipedia.org/wiki/Goertzel_algorithm>
    """

    cdef:
        double s
        double s_prev = 0
        double s_prev2 = 0
        double coeff = 2 * cos(2 * M_PI * (ft / fs))
        Py_ssize_t N = x.shape[0]
        Py_ssize_t ii

    for ii in range(N):
        s = x[ii] + (coeff * s_prev) - s_prev2
        s_prev2 = s_prev
        s_prev = s

    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2