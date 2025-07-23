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
                 choices=["gaussian"], help="type of convolution kernel")
CLI.add_argument("--kernel_params", nargs='+', type=str, required=True,
                 help="parameters of convolution kernel")


def gaussian_kernel(duration_ms, sampling_rate_Hz, std_dev_ms):
    """
    Generates a Gaussian kernel for smoothing.

    Parameters:
    - duration_ms: The duration over which the kernel is non-zero.
    - sampling_rate_Hz: The number of samples per second.
    - std_dev_ms: The standard deviation of the Gaussian kernel in ms.

    Returns:
    - Gaussian kernel array.
    """
    sampling_duration_s = 1 / sampling_rate_Hz
    sampling_duration_ms = sampling_duration_s * 1000.0  # ms / s
    kernel_start_ms = -0.5*duration_ms
    kernel_stop_ms = 0.5*duration_ms
    t_ms = np.arange(kernel_start_ms, kernel_stop_ms+0.5*sampling_duration_ms, sampling_duration_ms)
    gaussian = np.exp(-0.5 * (t_ms / std_dev_ms) ** 2)
    gaussian /= gaussian.sum()  # Normalize the kernel to ensure the signal energy is preserved
    return gaussian


def smooth_signal(asig, kernel_type, kernel_params):
    """
    Smooths the given signal by convolving it with the specified kernel.

    Parameters:
    - signal: The original signal
    - kernel: The smoothing kernel.

    Returns:
    - Smoothed signal.
    """

    if kernel_type=='gaussian':
        duration_ms = kernel_params['duration_ms']
        sampling_rate_Hz = kernel_params['sampling_rate_Hz']
        std_dev_ms = kernel_params['std_dev_ms']
        kernel = gaussian_kernel(duration_ms, sampling_rate_Hz, std_dev_ms)
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
