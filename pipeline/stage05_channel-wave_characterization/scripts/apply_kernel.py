"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage import convolve as conv

KernelDict = {"Simple": (np.array([[-0, 0, 0],[-1, 0, 1],[-0, 0, 0]], dtype=float) * 1/2,
                         np.array([[-0, -1, -0],[ 0,  0,  0],[ 0,  1,  0]], dtype=float) * 1/2,
                         (1,1)),
              "Sobol":  (np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=float) * 1/8,
                         np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]], dtype=float) * 1/8,
                         (1,1)),
              "Large": (np.array([[-1.1, -1.3, 0, 1.3, 1.1],[-1.3, -2.1, 0, 2.1, 1.3],
                                  [-1.5, -3.0, 0, 3.0, 1.5],[-1.3, -2.1, 0, 2.1, 1.3],
                                  [-1.1, -1.3, 0, 1.3, 1.1]], dtype=float) * 1/32.2,
                        np.array([[-1.1, -1.3, -1.5, -1.3, -1.1],[-1.3, -2.1, -3.0, -2.1, -1.3],
                                  [ 0,    0,    0,    0,    0],[ 1.3,  2.1,  3.0,  2.1, 1.3],
                                  [ 1.1,  1.3,  1.5,  1.3, 1.1]], dtype=float) * 1/32.2,
                        (2,2))}

def nanconv2d(frame, kernel, kernel_center=None):
    dx, dy = kernel.shape
    dimx, dimy = frame.shape
    dframe = np.empty((dimx, dimy))*np.nan

    if kernel_center is None:
        kernel_center = [int((dim-1)/2) for dim in kernel.shape]

    # inverse kernel to mimic behavior or regular convolution algorithm
    k = kernel[::-1, ::-1]
    ci = dx - 1 - kernel_center[0]
    cj = dy - 1 - kernel_center[1]

    # loop over each frame site
    for i,j in zip(*np.where(np.isfinite(frame))):
        site = frame[i,j]

        # loop over kernel window for frame site
        window = np.zeros((dx,dy), dtype=float)*np.nan
        for di,dj in itertools.product(range(dx), range(dy)):

            # kernelsite != 0, framesite within borders and != nan
            if k[di,dj] and 0 <= i+di-ci < dimx and 0 <= j+dj-cj < dimy \
                        and np.isfinite(frame[i+di-ci,j+dj-cj]):
                sign = -1*np.sign(k[di,dj])
                window[di,dj] = sign * (site - frame[i+di-ci,j+dj-cj])

        xi, yi = np.where(np.logical_not(np.isnan(window)))
        if np.sum(np.logical_not(np.isnan(window))) > dx*dy/10:
            dframe[i,j] = np.average(window[xi,yi], weights=abs(k[xi,yi]))
    return dframe

def convolve_triggers(triggers, kernel_name):
    kernelX = KernelDict[kernel_name][0]
    kernelY = KernelDict[kernel_name][1]
    return(nanconv2d(triggers, kernelX), nanconv2d(triggers, kernelY))



