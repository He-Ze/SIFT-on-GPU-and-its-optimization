import cv2
import ctypes
import numpy as np
from ctypes import cdll
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mg_cv = cv2.imread("test.JPG")
    gray=cv2.cvtColor(mg_cv,cv2.COLOR_BGR2GRAY)
    n = gray.shape[0]
    m = gray.shape[1]
    res = np.empty((gray.shape[0], gray.shape[1]), dtype='float64')
    gray = gray.astype('float64') # double 
    
    kr = 16
    ks = 4
    ko = 8
    test = cdll.LoadLibrary("./lib/siftdevice.so")
    S = 3
    sigma_init = 0.5
    contrast_threshold = 0.03
    edge_response_threshold = 10.0
    max_iterpolation = 10
    time_arr4 = np.empty(4, dtype='float64')
    
    test = cdll.LoadLibrary("./lib/siftdevice.so")
    test.sift.argtypes = [np.ctypeslib.ndpointer(dtype=gray.dtype, ndim=2, shape=gray.shape, flags='C_CONTIGUOUS'), 
                              np.ctypeslib.ndpointer(dtype=res.dtype, ndim=2, shape=res.shape, flags='C_CONTIGUOUS'),
                              ctypes.c_int,  # n
                              ctypes.c_int,  # m
                              ctypes.c_int,  # kr
                              ctypes.c_int,  # ks
                              ctypes.c_int,  # ko
                              ctypes.c_int,  # S
                              ctypes.c_double,  # sigma_init
                              ctypes.c_double,  # contrast_threshold
                              ctypes.c_double,  # edge_response_threshold
                              ctypes.c_int,  # max_iterpolation
                              np.ctypeslib.ndpointer(dtype=time_arr4.dtype, ndim=1, shape=time_arr4.shape)
                             ]
    test.sift(gray, res, n, m, kr, ks, ko, S, 
              sigma_init, contrast_threshold, 
              edge_response_threshold, max_iterpolation, time_arr4)
    plt.imshow(res, cmap='gray')
    plt.savefig("res/device.png")