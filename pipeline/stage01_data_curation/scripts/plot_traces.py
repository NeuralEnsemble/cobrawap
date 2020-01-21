import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import quantities as pq
import random
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--t_start", nargs='?', type=float)
    CLI.add_argument("--t_stop", nargs='?', type=float)
    CLI.add_argument("--channel", nargs='+', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        asig = io.read_block().segments[0].analogsignals

    check_analogsignal_shape(asig)
    asig = asig[0]

    dim_t, channel_num = asig.shape

    if not isinstance(args.channel, list):
        args.channel = [args.channel]
    for i, channel in enumerate(args.channel):
        if channel is None or channel >= channel_num:
            args.channel[i] = random.randint(0,channel_num)

    args.t_start = max([args.t_start, asig.t_start.rescale('s').magnitude])
    args.t_stop = min([args.t_stop, asig.t_stop.rescale('s').magnitude])
    asig = asig.time_slice(args.t_start*pq.s, args.t_stop*pq.s)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    offset = np.max(np.abs(asig.as_array()[:,args.channel]))

    for i, channel in enumerate(args.channel):
        ax.plot(asig.times, asig.as_array()[:,channel] + i*offset,
                label='channel {}'.format(channel))

    annotations = ''
    for key in asig.annotations.keys():
        if key not in ['nix_name', 'neo_name']:
            annotations += '{}: {}\n'.format(key, asig.annotations[key])
    array_annotations = ''
    for key in asig.array_annotations.keys():
        array_annotations += '{}: {}\n'.format(key, asig.array_annotations[key][args.channel[0]])

    ax.text(ax.get_xlim()[1]*1.02, ax.get_ylim()[0],
            'CHANNEL {} \n'.format(args.channel[0])\
            + '\n ANNOTATIONS:\n' + annotations \
            + '\n ARRAY ANNOTATIONS:\n' + array_annotations)

    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))
    ax.set_ylabel(asig.units)

    plt.legend()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output, bbox_inches='tight')
