"""
Convolves the signal of each channel.
"""

import argparse
import numpy as np
import os
from pathlib import Path
from utils.io_utils import load_neo, write_neo
from utils.parse import parse_string2dict
from scipy.signal import convolve

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--kernel_type", nargs='?', type=str, required=True,
                 choices=["gaussian", "biexponential"],
                 help="type of convolution kernel")
CLI.add_argument("--kernel_params", nargs='+', type=str, required=True,
                 help="parameters of convolution kernel")


def gaussian_kernel(sampling_rate_Hz, std_dev_ms, duration_ms=None):
    """
    Generates a Gaussian convolution kernel.

    Parameters:
    - sampling_rate_Hz:  The number of samples per second.
    - std_dev_ms:        The standard deviation of the Gaussian kernel in ms.
    - duration_ms:       [Optional] The duration over which the kernel is non-zero;
                         default value is 8 times the provided std_dev_ms,
                         suitably rounded to the closest multiple integer
                         of the sampling time step.

    Returns:
    - Gaussian kernel array.
    """

    sampling_dt_s = 1 / sampling_rate_Hz
    sampling_dt_ms = sampling_dt_s * 1000
    if duration_ms:
        half_width = 0.5*duration_ms
    else:
        # default span is 4 times std_dev_ms, on both sides
        half_width = 4*std_dev_ms
    n = math.ceil(half_width/sampling_dt_ms)
    times = np.linspace(-n*sampling_dt_ms, n*sampling_dt_ms, 2*n+1)

    kernel = np.exp(-0.5 * (times / std_dev_ms) ** 2)
    kernel /= np.sum(kernel)

    return kernel


def biexponential_kernel(sampling_rate_Hz, tau_rise_ms, tau_decay_ms, duration_ms=None):
    """
    Generates a bi-exponential convolution kernel.

    Parameters:
    - sampling_rate_Hz:  The number of samples per second.
    - tau_rise_ms:       Time constant for the rising phase in ms.
    - tau_decay_ms:      Time constant for the decaying phase in ms.
    - duration_ms:       [Optional] The duration over which the kernel is non-zero;
                         default value is 8 times the sum of the provided values
                         for tau_rise_ms and tau_decay_ms.

    Returns:
    - Biexponential kernel array.
    """

    sampling_dt_s = 1 / sampling_rate_Hz
    sampling_dt_ms = sampling_dt_s * 1000
    if duration_ms:
        half_width = 0.5*duration_ms
    else:
        # default span is 4 times (tau_rise_ms + tau_decay_ms), on both sides
        half_width = 4*(tau_rise_ms+tau_decay_ms)
    n = math.ceil(half_width/sampling_dt_ms)
    times = np.linspace(-n*sampling_dt_ms, n*sampling_dt_ms, 2*n+1)

    kernel = (np.exp(-times / tau_decay_ms) - np.exp(-times / tau_rise_ms)) * np.heaviside(times, 0)
    kernel /= np.sum(kernel)

    return kernel


def smooth_signal(asig, kernel_type, kernel_params):
    """
    Smooths the given signal by convolving it with the specified kernel.

    Parameters:
    - signal: The original signal.
    - kernel: The smoothing kernel.

    Returns:
    - Smoothed signal.
    """

    try:
        duration_ms = kernel_params['duration_ms']
    except KeyError:
        duration_ms = None

    if kernel_type=='gaussian':
        sampling_rate_Hz = kernel_params['sampling_rate_Hz']
        std_dev_ms = kernel_params['std_dev_ms']
        kernel = gaussian_kernel(sampling_rate_Hz, std_dev_ms, duration_ms)
    elif kernel_type=='biexponential':
        sampling_rate_Hz = kernel_params['sampling_rate_Hz']
        tau_rise_ms = kernel_params['tau_rise_ms']
        tau_decay_ms = kernel_params['tau_decay_ms']
        kernel = biexponential_kernel(sampling_rate_Hz, tau_rise_ms, tau_decay_ms, duration_ms)
    else:
        kernel = None

    convolved_signals = np.empty(asig.shape)
    convolved_signals.fill(np.nan)

    for channel in range(asig.shape[1]):
        channel_signal = asig.as_array()[:,channel]
        if np.isnan(channel_signal).any():
            continue
        convolved = convolve(channel_signal, kernel, mode='same')
        convolved_signals[:,channel] = convolved

    convolved_asig = asig.duplicate_with_new_data(convolved_signals)
    convolved_asig.array_annotate(**asig.array_annotations)

    return convolved_asig


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    kernel_params = parse_string2dict(args.kernel_params)
    convolved_asig = smooth_signal(asig, args.kernel_type, kernel_params)

    convolved_asig.description += "Convolved with a {} kernel of parameters {} ({}). "\
                        .format(args.kernel_type, args.kernel_params, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = convolved_asig

    write_neo(args.output, block)
