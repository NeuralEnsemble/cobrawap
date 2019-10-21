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
    CLI.add_argument("--MUA_data",   nargs='?', type=str)
    CLI.add_argument("--tstart", nargs='?', type=float)
    CLI.add_argument("--tstop",  nargs='?', type=float)
    CLI.add_argument("--channel",nargs='?', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        asig = io.read_block().segments[0].analogsignals[0]
    with neo.NixIO(args.MUA_data) as io:
        MUA_asig = io.read_block().segments[0].analogsignals[0]

    dim_t, channel_num = asig.shape

    if args.channel is None:
        args.channel = random.randint(0,channel_num)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    asig = zscore(asig.time_slice(args.tstart*pq.s, args.tstop*pq.s))
    ax.plot(asig.times, asig.as_array()[:,args.channel], label='original signal')

    MUA_asig = zscore(MUA_asig.time_slice(args.tstart*pq.s, args.tstop*pq.s))
    ax.plot(MUA_asig.times, MUA_asig.as_array()[:,args.channel]+5,
            label='MUA')

    # ToDo: add actual axis (left and right) for raw and MUA signal
    
    ax.set_title('Channel {}'.format(args.channel))
    ax.set_xlabel('time [s]')

    plt.legend()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output)
