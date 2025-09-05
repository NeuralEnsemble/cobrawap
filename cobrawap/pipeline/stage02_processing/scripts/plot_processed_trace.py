"""
Plot an example signal trace before and after application of some processing steps.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import os
from utils.io_utils import load_neo, save_plot
from utils.neo_utils import time_slice
from utils.parse import none_or_float

CLI = argparse.ArgumentParser()
CLI.add_argument("--original_data", nargs='?', type=Path, required=True,
                 help="path to original input data in neo format")
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to processed input data in neo format")
CLI.add_argument("--img_dir", nargs='?', type=Path, required=True,
                 help="path of output figure directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 default='processed_trace_channel0.png',
                 help='example filename for channel 0')
CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--t_stop", nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")
CLI.add_argument("--channels", nargs='+', type=int, default=0,
                 help="channel to plot")

def plot_traces(original_asig, processed_asig, channel):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax1 = plt.subplots()
    palette = sns.color_palette()

    ax1.plot(original_asig.times,
            original_asig.as_array()[:,channel],
            color=palette[0])
    ax1.set_ylabel('original signal', color=palette[0])
    ax1.tick_params('y', colors=palette[0])

    ax2 = ax1.twinx()
    ax2.plot(processed_asig.times,
            processed_asig.as_array()[:,channel],
            color=palette[1])
    ax2.set_ylabel('processed signal', color=palette[1])
    ax2.tick_params('y', colors=palette[1])

    ax1.set_title('Channel {}'.format(channel))
    ax1.set_xlabel('time [{}]'.format(original_asig.times.units.dimensionality.string))

    return ax1, ax2


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    orig_asig = load_neo(args.original_data, 'analogsignal', lazy=False)
    orig_asig = time_slice(orig_asig, t_start=args.t_start, t_stop=args.t_stop,
                           lazy=False, channel_indexes=args.channels)

    proc_asig = load_neo(args.data, 'analogsignal', lazy=False)
    proc_asig = time_slice(proc_asig, t_start=args.t_start, t_stop=args.t_stop,
                           lazy=False, channel_indexes=args.channels)

    for channel in args.channels:
        plot_traces(orig_asig, proc_asig, channel)
        output_path = os.path.join(args.img_dir,
                                   args.img_name.replace('_channel0', f'_channel{channel}'))
        save_plot(output_path)
