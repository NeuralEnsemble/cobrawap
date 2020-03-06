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
    CLI.add_argument("--signal", nargs='?', type=str)
    CLI.add_argument("--y", nargs='?', type=int)
    CLI.add_argument("--x", nargs='?', type=int)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--highcut", nargs='?', type=float)
    CLI.add_argument("--lowcut", nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.signal) as io:
        seg = io.read_block().segments[0]
        up_transitions = seg.spiketrains
        signal = seg.analogsignals[0]

    filt_signal = el.signal_processing.butter(signal.as_array()[:,args.x,args.y],
                                            highpass_freq=args.lowcut,
                                            lowpass_freq=args.highcut,
                                            order=2,
                                            fs=signal.sampling_rate)

    hilbert_signal = el.signal_processing.hilbert(signal)

    pixel_pos = signal.shape[2]*args.x + args.y

    sns.set(style='ticks', palette='Set2', context='talk')
    sns.set_color_codes()
    fig, ax = plt.subplots()

    ax.plot(signal.times,
            signal.as_array()[:,args.x,args.y],
            label=r'Ca$^+$ Signal')

    # ax.plot(signal.times,
    #         filt_signal,
    #         c='r',
    #         label='filtered signal [{}-{} Hz]'.format(args.lowcut, args.highcut))

    ax.plot(hilbert_signal.times,
            np.angle(hilbert_signal.as_array()[:,args.x,args.y])/np.pi,
            color = 'k',
            label=r'Hilbert Phase [$\pi$]')

    ax.plot(up_transitions[pixel_pos],
            np.zeros_like(up_transitions[pixel_pos]), color='r',
            linestyle='None', marker='D', markersize=5, label='Up Transitions')

    ax.set_xlabel('time [s]')
    # ax.set_ylabel(r'Ca$^+$ signal')
    ax.set_title('signal of pixel ({},{})'.format(args.x,args.y))
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
