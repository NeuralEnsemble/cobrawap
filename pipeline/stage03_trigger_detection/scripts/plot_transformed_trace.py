import matplotlib.pyplot as plt
import seaborn as sns
from elephant.signal_processing import zscore
import numpy as np
import argparse
import os
import quantities as pq
import neo
import random


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--data",   nargs='?', type=str)
    CLI.add_argument("--trans_data",   nargs='?', type=str)
    CLI.add_argument("--t_start", nargs='?', type=float)
    CLI.add_argument("--t_stop",  nargs='?', type=float)
    CLI.add_argument("--channel",nargs='?', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        asig = io.read_block().segments[0].analogsignals[0]
    with neo.NixIO(args.trans_data) as io:
        trans_asig = io.read_block().segments[0].analogsignals[0]

    dim_t, channel_num = asig.shape

    if args.channel is None or args.channel <= channel_num:
        args.channel = random.randint(0,channel_num)

    args.t_start = max([args.t_start, asig.t_start.rescale('s').magnitude])
    args.t_stop = max([args.t_stop, asig.t_stop.rescale('s').magnitude])

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    asig = zscore(asig.time_slice(args.t_start*pq.s, args.t_stop*pq.s))
    ax.plot(asig.times, asig.as_array()[:,args.channel], label='original signal')

    trans_asig = zscore(trans_asig.time_slice(args.t_start*pq.s, args.t_stop*pq.s))
    ax.plot(trans_asig.times, trans_asig.as_array()[:,args.channel]+8,
            label='transformed signal')

    # ToDo: add actual axis (left and right) for raw and logMUA signal

    ax.set_title('Channel {}'.format(args.channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))

    plt.legend()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output)
