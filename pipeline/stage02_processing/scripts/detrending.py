"""
ToDo
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import neo
import argparse
import os
from copy import copy
import warnings
from utils.io import load_neo, write_neo, save_plot

## REPLACED BY SCIPY FUNCTION
# def detrending(signal, order):
#     if isinstance(signal, neo.AnalogSignal):
#         X = signal.as_array()
#     elif isinstance(signal, pq.quantity.Quantity):
#         X = copy(signal.magnitude)
#     elif isinstance(signal, np.ndarray):
#         X = copy(signal)
#     else:
#         raise TypeError('Input signal must be either an AnalogSignal,'
#                       + 'a quantity array, or a numpy array.')
#     if not type(order) == int:
#         raise TypeError("Detrending order needs to be int.")
#     if order < 0 or order > 4:
#         raise InputError("Detrending order must be between 0 and 4!")
#
#     dim_t, num_channels = signal.shape
#     window_size = len(signal)
#
#     if order > 0:
#         X = X - np.mean(X, axis=0)
#     if order > 1:
#         factor = [1, 1/2., 1/6.]
#         for i in np.arange(2, order):
#             detrend = np.zeros_like(X)
#             for channel, x in enumerate(X.T):
#                 detrend[:,channel] =\
#                     np.linspace(-window_size/2., window_size/2., window_size)**i \
#                     * np.mean(np.diff(x, n=i)) * factor[i-2]
#             X = X - detrend
#
#     if isinstance(signal, neo.AnalogSignal):
#         signal_out = signal.duplicate_with_new_data(X)
#         signal_out.array_annotate(**signal.array_annotations)
#         return signal_out
#     elif isinstance(signal, pq.quantity.Quantity):
#         return X * signal.units
#     elif isinstance(signal, np.ndarray):
#         return X

def detrend(asig, order):
    if (args.order != 0) and (args.order != 1):
        warnings.warn("Detrending order must be either 0 (constant) or 1 (linear)! Skip.")
        return asig

    dtrend = 'linear' if args.order else 'constant'
    detrended_signals = np.empty(asig.shape)
    detrended_signals.fill(np.nan)

    for channel in range(asig.shape[1]):
        channel_signal = asig.as_array()[:,channel]
        if np.isnan(channel_signal).any():
            continue
        detrended = scipy.signal.detrend(channel_signal, type=dtrend, axis=0)
        detrended_signals[:,channel] = detrended
    detrend_asig = asig.duplicate_with_new_data(detrended_signals)
    detrend_asig.array_annotate(**asig.array_annotations)
    return detrend_asig


def plot_detrend(asig, detrend_asig, channel):
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(17,8))

    ax[0].plot(asig.times, asig.as_array()[:,channel], color='b', linewidth=1)
    ax[0].axhline(0, linestyle="--", color='k', linewidth=1)
    ax[0].set_ylabel('signal')

    ax[1].plot(asig.times, detrend_asig.as_array()[:,channel], color='g', linewidth=1)
    ax[1].axhline(0, linestyle="--", color='k', linewidth=1)
    ax[1].set_ylabel('detrended signal')
    ax[1].set_xlabel(f'time [{asig.times.dimensionality.string}]')
    return ax


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--order", nargs='?', type=int, default=1,
                     help="detrending order")
    CLI.add_argument("--img_dir",  nargs='?', type=str, required=True,
                     help="path of output figure directory")
    CLI.add_argument("--img_name", nargs='?', type=str,
                     default='processed_trace_channel0.png',
                     help='example filename for channel 0')
    CLI.add_argument("--channels", nargs='+', type=int, default=0,
                     help="channel to plot")
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    detrend_asig = detrend(asig, args.order)

    for channel in args.channels:
        plot_detrend(asig, detrend_asig, channel)
        output_path = os.path.join(args.img_dir,
                                   args.img_name.replace('_channel0', f'_channel{channel}'))
        save_plot(output_path)

    detrend_asig.name += ""
    detrend_asig.description += "Detrended by order {} ({}). "\
                        .format(args.order, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = detrend_asig

    write_neo(args.output, block)
