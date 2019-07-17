import neo
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--preprocessed_signal", nargs='?', type=str)
    CLI.add_argument("--clean_signal", nargs='?', type=str)
    CLI.add_argument("--y", nargs='?', type=int)
    CLI.add_argument("--x", nargs='?', type=int)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--highcut", nargs='?', type=float)
    CLI.add_argument("--lowcut", nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.preprocessed_signal) as io:
        preprocessed_signal = io.read_block().segments[0].analogsignals[0]

    with neo.NixIO(args.clean_signal) as io:
        clean_signal_seg = io.read_block().segments[0]
    clean_analogsignal = clean_signal_seg.analogsignals[0]
    up_transitions = clean_signal_seg.spiketrains

    pixel_pos = clean_analogsignal.shape[2]*args.x + args.y

    sns.set(style='ticks', palette='Set2', context='paper')
    sns.set_color_codes()
    fig, ax = plt.subplots()

    ax.plot(preprocessed_signal.times,
            preprocessed_signal.as_array()[:,args.x,args.y],
            c='b', label='preprocessed_signal')

    ax.plot(clean_analogsignal.times,
            clean_analogsignal.as_array()[:,args.x,args.y],
            c='r',
            label='clean signal [{}-{} Hz]'.format(args.lowcut, args.highcut))

    ax.plot(up_transitions[pixel_pos],
            np.zeros_like(up_transitions[pixel_pos]),
            linestyle='None', marker='.', color='k', label='minima')

    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'Ca$^+$ signal')
    ax.set_title('signal of pixel ({},{})'.format(args.x,args.y))
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
