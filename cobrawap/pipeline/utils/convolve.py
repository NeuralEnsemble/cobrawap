import numpy as np
import itertools
import warnings
from scipy.ndimage import convolve as conv
from types import SimpleNamespace

simple_3x3 = SimpleNamespace(
             x=np.array([[-0, 0, 0],
                         [-1, 0, 1],
                         [-0, 0, 0]], dtype=float),
             y=np.array([[-0, -1, -0],
                         [ 0,  0,  0],
                         [ 0,  1,  0]], dtype=float))

prewitt_3x3 = SimpleNamespace(
             x=np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=float),
             y=np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=float))

scharr_3x3 = SimpleNamespace(
             x=np.array([[ -3, 0,  3],
                         [-10, 0, 10],
                         [ -3, 0,  3]], dtype=float),
             y=np.array([[-3, -10, -3],
                         [ 0,   0,  0],
                         [ 3,  10,  3]], dtype=float))

sobel_3x3 = SimpleNamespace(
            x=np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float),
            y=np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float))

sobel_5x5 = SimpleNamespace(
            x=np.array([[ -5, -4, 0,  4,  5],
                        [ -8,-10, 0, 10,  8],
                        [-10,-20, 0, 20, 10],
                        [ -8,-10, 0, 10,  8],
                        [ -5, -4, 0,  4,  5]], dtype=float),
            y=np.array([[ -5, -8, -10, -8, -5],
                        [ -4,-10, -20,-10,  4],
                        [  0,  0,   0,  0,  0],
                        [  4, 10,  20, 10,  4],
                        [  5,  8,  10,  8,  5]], dtype=float))

sobel_7x7 = SimpleNamespace(
            x=np.array([[-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18],
                        [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                        [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                        [-3/9 , -2/4 , -1/1 , 0,  1/1 , 2/4 , 3/9 ],
                        [-3/10, -2/5 , -1/2 , 0,  1/2 , 2/5 , 3/10],
                        [-3/13, -2/8 , -1/5 , 0,  1/5 , 2/8 , 3/13],
                        [-3/18, -2/13, -1/10, 0,  1/10, 2/13, 3/18]], dtype=float),
            y=np.array([[-3/18, -3/13, -3/10, -3/9,  -3/10, -3/13, -3/18],
                        [-2/13, -2/8 , -2/5 , -2/4,  -2/5 , -2/8 , -2/13],
                        [-1/10, -1/5 , -1/2 , -1/1,  -1/2 , -1/5 , -1/10],
                        [    0,    0,     0 ,    0,     0 ,    0 ,     0],
                        [ 1/10,  1/5 ,  1/2 ,  1/1,   1/2 ,  1/5 ,  1/10],
                        [ 2/13,  2/8 ,  2/5 ,  2/4,   2/5 ,  2/8 ,  2/13],
                        [ 3/18,  3/13,  3/10,  3/9,   3/10,  3/13,  3/18]], dtype=float))

for kernel in [simple_3x3, prewitt_3x3, scharr_3x3,
               sobel_3x3, sobel_5x5, sobel_7x7]:
    kernel.x /= np.sum(np.abs(kernel.x))
    kernel.y /= np.sum(np.abs(kernel.y))

def get_kernel(kernel_name):
    if kernel_name.lower() in ['simple_3x3', 'simple']:
        return simple_3x3
    elif kernel_name.lower() in ['prewitt_3x3', 'prewitt']:
        return prewitt_3x3
    elif kernel_name.lower() in ['scharr_3x3', 'scharr']:
        return scharr_3x3
    elif kernel_name.lower() in ['sobel_3x3', 'sobel']:
        return sobel_3x3
    elif kernel_name.lower() in ['sobel_5x5']:
        return sobel_5x5
    elif kernel_name.lower() in ['sobel_7x7']:
        return sobel_7x7
    else:
        warnings.warn(f'Deriviative name {kernel_name} is not implemented, '
                     + 'using sobel filter instead.')
        return sobel_3x3


def nan_conv2d(frame, kernel, kernel_center=None):
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

norm_angle = lambda p: -np.mod(p + np.pi, 2*np.pi) + np.pi

def phase_conv2d(frame, kernel, kernel_center=None):
    dx, dy = kernel.shape
    dimx, dimy = frame.shape
    dframe = np.zeros_like(frame)

    if kernel_center is None:
        kernel_center = [int((dim-1)/2) for dim in kernel.shape]

    # inverse kernel to mimic behavior or regular convolution algorithm
    k = kernel[::-1, ::-1]
    ci = dx - 1 - kernel_center[0]
    cj = dy - 1 - kernel_center[1]

    # loop over kernel window for each frame site
    for i,j in zip(*np.where(np.isfinite(frame))):
        phase = frame[i,j]
        dphase = np.zeros((dx,dy), dtype=float)

        for di,dj in itertools.product(range(dx), range(dy)):

            # kernelsite != 0, framesite within borders and != nan
            if k[di,dj] and 0 <= i+di-ci < dimx and 0 <= j+dj-cj < dimy \
            and np.isfinite(frame[i+di-ci,j+dj-cj]):
                sign = -1*np.sign(k[di,dj])
                # pos = clockwise from phase to frame[..]
                dphase[di,dj] = sign*norm_angle(phase-frame[i+di-ci,j+dj-cj])

        if dphase.any():
            dframe[i,j] = np.average(dphase, weights=abs(k)) / np.pi
    return dframe
