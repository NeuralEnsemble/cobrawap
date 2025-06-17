"""
Offsets the signal in all channels by a fixed value.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_int, none_or_float, none_or_path
from utils.neo_utils import time_slice

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--offset", nargs='?', type=none_or_float, default=None,
                 help="offset the signal by some value")
CLI.add_argument("--img_dir", nargs='?', type=none_or_path, default=None,
                 help="path of figure directory")
CLI.add_argument("--img_name", nargs='?', type=str, default='offset_channel0.png',
                 help='example image filename for channel 0')
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="plotting start time in seconds")
CLI.add_argument("--plot_tstop",  nargs='?', type=none_or_float, default=10,
                 help="plotting stop time in seconds")

def offset_signal(asig, offset=None):
    if offset is None:
        offset = 0

    new_signal = asig.as_array() + offset

    new_asig = asig.duplicate_with_new_data(new_signal)

    new_asig.array_annotate(**asig.array_annotations)
    new_asig.annotate(offset=offset)
    new_asig.description += f"Offset by {offset_signal} ({__file__}). "

    return new_asig


def plot_signal(asig, new_asig, channel=0, t_start=None, t_stop=None):
    fig, ax = plt.subplots(figsize=(17,8))

    asig = time_slice(asig, t_start=t_start, t_stop=t_stop)
    ax.plot(asig.times, asig.as_array()[:,channel], color='b',
            linewidth=1, label='original signal')

    new_asig = time_slice(new_asig, t_start=t_start, t_stop=t_stop)
    ax.plot(new_asig.times, new_asig.as_array()[:,channel], color='g',
            linewidth=1, label='offset signal')

    ax.set_xlabel(f'time [{asig.times.dimensionality.string}]')
    ax.set_ylabel('signal')
    plt.legend()
    return None


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    # LOADING
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    # PERFORMING METHOD
    new_asig = offset_signal(asig, offset=args.offset)
    block.segments[0].analogsignals[0] = new_asig

    # PLOTTING
    if args.plot_channels[0] is not None:
        if args.img_dir is None:
            args.img_dir = args.output.parent
        for channel in args.plot_channels:
            plot_signal(asig, new_asig, channel=channel,
                        t_start=args.plot_tstart, t_stop=args.plot_tstop)
            output_path = args.img_dir \
                        / args.img_name.replace('_channel0', f'_channel{channel}')
            save_plot(output_path)

    # SAVING
    write_neo(args.output, block)
