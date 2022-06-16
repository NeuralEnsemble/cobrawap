"""
Plot traces
-----------

description...

Input: neo.Block with ...

Output: neo.Block + ...

"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import quantities as pq
import random
from utils.io import load_neo, save_plot
from utils.neo import time_slice
from utils.parse import parse_plot_channels, none_or_int, determine_dims


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

    coords = np.stack((asig.array_annotations['x_coords'],
                       asig.array_annotations['y_coords']), axis=-1)

    ax.text(ax.get_xlim()[1]*1.05, ax.get_ylim()[0],
            f'ANNOTATIONS FOR CHANNEL(s) {channels} \n'\
          +  '\n ANNOTATIONS:\n' + '\n'.join(annotations) \
          +  '\n\n ARRAY ANNOTATIONS:\n' + '\n'.join(array_annotations) +'\n' \
          + f' t_start: {asig.t_start}; t_stop: {asig.t_stop} \n' \
          + f' dimensions(x,y): {determine_dims(coords)}')

    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'channels [in {asig.units.dimensionality.string}]')
    ax.set_yticks([i*offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    return ax


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output figure")
    CLI.add_argument("--t_start", nargs='?', type=float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=float, default=10,
                     help="stop time in seconds")
    CLI.add_argument("--channels", nargs='+', type=none_or_int, default=0,
                     help="list of channels to plot")
    args = CLI.parse_args()

    asig = load_neo(args.data, 'analogsignal', lazy=True)

    channels = parse_plot_channels(args.channels, args.data)

    asig = time_slice(asig, t_start=args.t_start, t_stop=args.t_stop,
                      lazy=True, channel_indexes=channels)

    ax = plot_traces(asig, channels)
    save_plot(args.output)
