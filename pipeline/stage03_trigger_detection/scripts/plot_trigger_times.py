import numpy as np
from elephant.signal_processing import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import quantities as pq
import argparse
import os
import random

def plot_states(times, labels, ax, tstart, tstop, label=''):
    if labels[0].decode('UTF-8') == 'DOWN':
        ax.axvspan(tstart, times[0], alpha=0.5, color='red')
    if labels[-1].decode('UTF-8') == 'UP':
        ax.axvspan(times[-1], tstop, alpha=0.5, color='red')

    for i, (time, label) in enumerate(zip(times, labels)):
        if label.decode('UTF-8') == 'UP' and i < len(times)-1:
            ax.axvspan(time, times[i+1], alpha=0.5, color='red',
                       label=label.decode('UTF-8') if not i else '')
    return None


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",        nargs='?', type=str)
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--tstart",        nargs='?', type=float)
    CLI.add_argument("--tstop",         nargs='?', type=float)
    CLI.add_argument("--channel",       nargs='?', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()


    asig = block.segments[0].analogsignals[0]
    asig = asig.time_slice(args.tstart*pq.s, args.tstop*pq.s)
    event = [evt for evt in block.segments[0].events if evt.name=='Transitions'][0]
    event = event.time_slice(args.tstart*pq.s, args.tstop*pq.s)

    dim_t, dim_channels = asig.shape

    if args.channel is None:
        args.channel = random.randint(0, dim_channels)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(asig.times, asig.as_array()[:,args.channel], label='signal')

    times = [time for i, time in enumerate(event.times)
             if event.array_annotations['channels'][i]==args.channel]
    labels = [label for i, label in enumerate(event.labels)
             if event.array_annotations['channels'][i]==args.channel]

    if 'DOWN'.encode('UTF-8') in labels:
        # plot up states
        plot_states(times, labels, ax, tstart=args.tstart, tstop=args.tstop,
                    label='UP states')
    elif 'UP'.encode('UTF-8') in labels:
        # plot only up transitions
        for i, trans_time in enumerate(times):
            ax.axvline(trans_time, c='k',
                       label='UP transitions' if not i else '')
    else:
        raise ValueError("No 'UP' (or 'DOWN') transition events found")

    ax.set_title('Channel {}'.format(args.channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))

    plt.legend()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output)
