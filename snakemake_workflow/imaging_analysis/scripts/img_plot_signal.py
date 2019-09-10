import neo
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import elephant as el
import scipy


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

    up_transitions = clean_signal_seg.spiketrains

    filt_signal = el.signal_processing.butter(preprocessed_signal.as_array()[:,args.x,args.y],
                                            highpass_freq=args.lowcut,
                                            lowpass_freq=args.highcut,
                                            order=2,
                                            fs=preprocessed_signal.sampling_rate)

    hilbert_signal = el.signal_processing.hilbert(preprocessed_signal)

    pixel_pos = preprocessed_signal.shape[2]*args.x + args.y

    sns.set(style='ticks', palette='Set2', context='paper')
    sns.set_color_codes()
    fig, ax = plt.subplots()

    ax.plot(preprocessed_signal.times,
            preprocessed_signal.as_array()[:,args.x,args.y],
            c='b', label='preprocessed_signal')

    ax.plot(preprocessed_signal.times,
            filt_signal,
            c='r',
            label='filtered signal [{}-{} Hz]'.format(args.lowcut, args.highcut))

    ax.plot(hilbert_signal.times,
            np.angle(hilbert_signal.as_array()[:,args.x,args.y])/3.,
            c='k',
            label='Phase Hilbert')

    ax.plot(up_transitions[pixel_pos],
            np.zeros_like(up_transitions[pixel_pos]),
            linestyle='None', marker='.', color='k', label='minima')

    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'Ca$^+$ signal')
    ax.set_title('signal of pixel ({},{})'.format(args.x,args.y))
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
