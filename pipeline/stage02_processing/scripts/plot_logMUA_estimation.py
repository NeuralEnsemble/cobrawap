import numpy as np
from elephant.signal_processing import zscore, butter
import neo
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",        nargs='?', type=str)
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--logMUA_data",   nargs='?', type=str)
    CLI.add_argument("--highpass_freq", nargs='?', type=float)
    CLI.add_argument("--lowpass_freq",  nargs='?', type=float)
    CLI.add_argument("--t_start",   nargs='?', type=float)
    CLI.add_argument("--t_stop",   nargs='?', type=float)
    CLI.add_argument("--channel",   nargs='?', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()
    with neo.NixIO(args.logMUA_data) as io:
        logMUA_block = io.read_block()

    asig = block.segments[0].analogsignals[0]

    dim_t, channel_num = asig.shape

    if args.channel is None or args.channel >= channel_num:
        print(args.channel, channel_num)
        args.channel = random.randint(0, channel_num-1)

    args.t_start = max([args.t_start, asig.t_start.rescale('s').magnitude])
    args.t_stop = min([args.t_stop, asig.t_stop.rescale('s').magnitude])

    signal = asig.time_slice(t_start=args.t_start*pq.s, t_stop=args.t_stop*pq.s)
    logMUA_signal = logMUA_block.segments[0].analogsignals[0]\
                    .time_slice(t_start=args.t_start*pq.s, t_stop=args.t_stop*pq.s)

    filt_signal = butter(signal,
                        highpass_freq=args.highpass_freq*pq.Hz,
                        lowpass_freq=args.lowpass_freq*pq.Hz)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(signal.times, zscore(signal).as_array()[:,args.channel],
            label='original signal')

    ax.plot(signal.times, zscore(filt_signal).as_array()[:,args.channel] + 10,
            label='signal [{}-{}Hz]'.format(args.highpass_freq, args.lowpass_freq),
            alpha=0.5)

    ax.plot(logMUA_signal.times, zscore(logMUA_signal).as_array()[:,args.channel] + 20,
            label='logMUA')

    ax.set_title('Channel {}'.format(args.channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))

    plt.legend()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output)
