import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import quantities as pq
import random
from utils import load_neo, save_plot, none_or_int


def plot_traces(asig, channels):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    offset = np.max(np.abs(asig.as_array()[:,channels]))

    for i, signal in enumerate(asig.as_array()[:,channels].T):
        ax.plot(asig.times, signal + i*offset)

    annotations = [f'{k}: {v}' for k,v in asig.annotations.items()]
    array_annotations = [f'{k}: {v[channels[0]]}'
                         for k,v in asig.array_annotations.items()]

    ax.text(ax.get_xlim()[1]*1.02, ax.get_ylim()[0],
            f'ANNOTATIONS FOR CHANNEL {channels[0]} \n'\
            + '\n ANNOTATIONS:\n' + '\n'.join(annotations) \
            + '\n\n ARRAY ANNOTATIONS:\n' + '\n'.join(array_annotations))

    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'channels [in {asig.units.dimensionality.string}]')
    ax.set_yticks([i*offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    return fig


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
    CLI.add_argument("--channel", nargs='+', type=none_or_int, default='None',
                     help="list of channels to plot")
    args = CLI.parse_args()
    CLI.print_help()

    asig = load_neo(args.data, 'analogsignal')

    # parsing plotting channels
    dim_t, channel_num = asig.shape
    for i, channel in enumerate(args.channel):
        if channel is None or channel >= channel_num:
            args.channel[i] = random.randint(0,channel_num)

    # slicing signals
    args.t_start = max([args.t_start, asig.t_start.rescale('s').magnitude])
    args.t_stop = min([args.t_stop, asig.t_stop.rescale('s').magnitude])
    asig = asig.time_slice(args.t_start*pq.s, args.t_stop*pq.s)

    fig = plot_traces(asig, args.channel)

    save_plot(args.output)
