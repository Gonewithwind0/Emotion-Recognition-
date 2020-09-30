import numpy as np
import scipy as sp

from scipy import signal

def MedianFilter(data,windowLength):
    return sp.signal.medfilt(data,windowLength)
