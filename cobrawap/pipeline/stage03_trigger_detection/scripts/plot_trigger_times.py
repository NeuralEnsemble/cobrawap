"""
Plot an excerpt of the input data and corresponding trigger times.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import os
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_int, none_or_float
from utils.neo_utils import time_slice

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path,
                 required=True, help="path of output directory")
CLI.add_argument("--filename", nargs='?', type=str,
                 default='trigger_times_channel0.png',
                 help='example filename for channel 0')
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--plot_tstop", nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")

def plot_trigger_times(asig, event, channel):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(asig.times, asig.as_array()[:,channel], label='signal')

    times = [time for i, time in enumerate(event.times)
             if event.array_annotations['channels'][i]==channel]
    labels = [label for i, label in enumerate(event.labels)
             if event.array_annotations['channels'][i]==channel]

    if 'DOWN'.encode('UTF-8') in labels or 'DOWN' in labels:
        # plot up states
        plot_states(times, labels, ax,
                    t_start=asig.t_start, t_stop=asig.t_stop, label='UP states')
    elif 'UP'.encode('UTF-8') in labels or 'UP' in labels:
        # plot only up transitions
        for i, trans_time in enumerate(times):
            ax.axvline(trans_time, c='k',
                       label='UP transitions' if not i else '')
    else:
        print("Warning: No 'UP' (or 'DOWN') transition events "\
            + f"in channel {channel} found!")

    ax.set_title(f'Channel {channel}')
    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'signal [{asig.units.dimensionality.string}]')

    plt.legend()
    return fig


def plot_states(times, labels, ax, t_start, t_stop, label=''):
    if labels[0] == 'DOWN'.encode('UTF-8') or labels[0] == 'DOWN':
        ax.axvspan(t_start, times[0], alpha=0.5, color='red')
    if labels[-1] == 'UP'.encode('UTF-8') or labels[-1] == 'UP':
        ax.axvspan(times[-1], t_stop, alpha=0.5, color='red')

    for i, (time, label) in enumerate(zip(times, labels)):
        if (label == 'UP'.encode('UTF-8') or label == 'UP') \
            and i < len(times)-1:
            ax.axvspan(time, times[i+1], alpha=0.5, color='red',
                       label='UP' if not i else '')
    return None


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    args.plot_tstart = asig.t_start if args.plot_tstart is None else args.plot_tstart
    args.plot_tstop = asig.t_stop if args.plot_tstop is None else args.plot_tstop
    # slice signals
    asig = time_slice(asig, args.plot_tstart, args.plot_tstop)

    # get transition events
    event = block.filter(name='transitions', objects="Event")[0]
    event = event.time_slice(args.plot_tstart, args.plot_tstop)

    for channel in args.plot_channels:
        plot_trigger_times(asig=asig,
                           event=event,
                           channel=channel)
        output_path = os.path.join(args.output,
                                   args.filename.replace('_channel0', f'_channel{channel}'))
        save_plot(output_path)
