"""
Detrend the signal in each channel by order 0 (constant) or 1 (linear).
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import warnings
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_int

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--order", nargs='?', type=int, default=1,
                 help="detrending order")
CLI.add_argument("--img_dir", nargs='?', type=Path, required=True,
                 help="path of output figure directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 default='processed_trace_channel0.png',
                 help='example filename for channel 0')
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")

def detrend(asig, order):
    if (order != 0) and (order != 1):
        warnings.warn("Detrending order must be either 0 (constant) or 1 (linear)! Skip.")
        return asig

    dtrend = 'linear' if order else 'constant'
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
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    detrend_asig = detrend(asig, args.order)

    if args.plot_channels[0] is not None:
        for channel in args.plot_channels:
            plot_detrend(asig, detrend_asig, channel)
            output_path = os.path.join(args.img_dir,
                                       args.img_name.replace('_channel0', f'_channel{channel}'))
            save_plot(output_path)

    detrend_asig.description += "Detrended by order {} ({}). "\
                        .format(args.order, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = detrend_asig

    write_neo(args.output, block)
