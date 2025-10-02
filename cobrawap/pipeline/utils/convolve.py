import itertools
import numpy as np
from scipy.ndimage import convolve as conv
from scipy.signal import convolve2d
from types import SimpleNamespace
import warnings

simple_3x3 = SimpleNamespace(
    x=np.array([[-0, 0, 0],
                [-1, 0, 1],
                [-0, 0, 0]], dtype=float),
    y=np.array([[-0, -1, -0],
                [0, 0, 0],
                [0, 1, 0]], dtype=float))

prewitt_3x3 = SimpleNamespace(
    x=np.array([[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]], dtype=float),
    y=np.array([[-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]], dtype=float))

scharr_3x3 = SimpleNamespace(
    x=np.array([[-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3]], dtype=float),
    y=np.array([[-3, -10, -3],
                [0, 0, 0],
                [3, 10, 3]], dtype=float))

sobel_3x3 = SimpleNamespace(
    x=np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=float),
    y=np.array([[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]], dtype=float))

sobel_5x5 = SimpleNamespace(
    x=np.array([[-5, -4, 0, 4, 5],
                [-8, -10, 0, 10, 8],
                [-10, -20, 0, 20, 10],
                [-8, -10, 0, 10, 8],
                [-5, -4, 0, 4, 5]], dtype=float),
    y=np.array([[-5, -8, -10, -8, -5],
                [-4, -10, -20, -10, 4],
                [0, 0, 0, 0, 0],
                [4, 10, 20, 10, 4],
                [5, 8, 10, 8, 5]], dtype=float))

sobel_7x7 = SimpleNamespace(
    x=np.array([[-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18],
                [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                [-3 / 9, -2 / 4, -1 / 1, 0, 1 / 1, 2 / 4, 3 / 9],
                [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                [-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18]], dtype=float),
    y=np.array([[-3 / 18, -3 / 13, -3 / 10, -3 / 9, -3 / 10, -3 / 13, -3 / 18],
                [-2 / 13, -2 / 8, -2 / 5, -2 / 4, -2 / 5, -2 / 8, -2 / 13],
                [-1 / 10, -1 / 5, -1 / 2, -1 / 1, -1 / 2, -1 / 5, -1 / 10],
                [0, 0, 0, 0, 0, 0, 0],
                [1 / 10, 1 / 5, 1 / 2, 1 / 1, 1 / 2, 1 / 5, 1 / 10],
                [2 / 13, 2 / 8, 2 / 5, 2 / 4, 2 / 5, 2 / 8, 2 / 13],
                [3 / 18, 3 / 13, 3 / 10, 3 / 9, 3 / 10, 3 / 13, 3 / 18]], dtype=float))

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
        warnings.warn(f'Derivative name {kernel_name} is not implemented, '
                      + 'using sobel filter instead.')
        return sobel_3x3


def nan_conv2d(image, kernel):
    mask = 1 - np.isnan(image)
    image_ = np.nan_to_num(image)
    conve = convolve2d(image_, kernel, mode='same')
    sum_ker = convolve2d(mask, kernel, mode='same')
    pro = np.multiply(sum_ker, image)
    quot = convolve2d(mask, abs(kernel), mode='same')
    conve = np.divide((conve - pro), quot)
    conve[np.isnan(image)] = np.nan
    return conve


def norm_angle(p):
    # maps angle to [-pi, pi]
    return -np.mod(p + np.pi, 2 * np.pi) + np.pi


def phase_conv2d(image, kernel):
    mask = 1 - np.isnan(image)
    image = np.nan_to_num(image)
    filters = []
    dx, dy = kernel.shape
    ci, cj = [int((dim - 1) / 2) for dim in kernel.shape]
    for di, dj in itertools.product(range(dx), range(dy)):
        if (di, dj) != (ci, cj):
            a = np.zeros((dx, dy))
            a[di, dj] = 1
            filters.append(a)
    n_channels = len(filters)
    dimx, dimy = image.shape
    conv = np.zeros((n_channels, dimx, dimy))
    for i in range(n_channels):
        channel = convolve2d(image, filters[i], mode='same')
        mean_ = convolve2d(mask, filters[i], mode='same') * image
        conv[i] = channel - mean_
    conv = norm_angle(conv)
    for i in range(n_channels):
        conv[i] = conv[i] * np.sum(filters[i] * kernel)
    conv = np.sum(conv, axis=0)
    quot = np.sum(abs(kernel)) * np.pi
    out = conv / quot
    return out * mask
