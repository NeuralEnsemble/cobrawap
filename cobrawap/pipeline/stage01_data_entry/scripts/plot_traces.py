"""
Plots excerpts of the input data with its corresponding metadata.
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io_utils import load_neo, save_plot
from utils.neo_utils import time_slice
from utils.parse import parse_plot_channels, none_or_int, none_or_float

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output figure")
CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--t_stop", nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")
CLI.add_argument("--channels", nargs='+', type=none_or_int, default=0,
                 help="list of channels to plot")

def plot_traces(asig, channels):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    offset = np.max(np.abs(asig.as_array()[:,channels]))
    for i, signal in enumerate(asig.as_array()[:,channels].T):
        ax.plot(asig.times, signal + i*offset)

    annotations = [f'{k}: {v}' for k,v in asig.annotations.items()
                               if k not in ['nix_name', 'neo_name']]
    array_annotations = [f'{k}: {v[channels]}'
                        for k,v in asig.array_annotations.items()]

    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1

    ax.text(1.05, 0.5,
            f'ANNOTATIONS FOR CHANNEL(s): {channels}' + '\n' \
            + '\n' \
            + 'ANNOTATIONS:' + '\n' \
            + ' - ' + '\n - '.join(annotations) + '\n' \
            + '\n' \
            + 'ARRAY ANNOTATIONS:' + '\n' \
            + ' - ' + '\n - '.join(array_annotations) + '\n' \
            + f' - t_start: {asig.t_start}; t_stop: {asig.t_stop}' + '\n' \
            + f' - dimensions(x,y): {dim_x}, {dim_y}',
            ha='left', va='center', transform=ax.transAxes)

    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'channels [in {asig.units.dimensionality.string}]')
    ax.set_yticks([i*offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    asig = load_neo(args.data, 'analogsignal', lazy=True)

    channels = parse_plot_channels(args.channels, args.data)

    asig = time_slice(asig, t_start=args.t_start, t_stop=args.t_stop,
                      lazy=True, channel_indexes=channels)

    ax = plot_traces(asig, channels)
    save_plot(args.output)
