"""
Calculate the period between two consecutive waves for each wave and channel.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path, none_or_str
from utils.neo_utils import analogsignal_to_imagesequence

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--kernel", "--KERNEL", nargs='?', type=none_or_str, default=None,
                 help="derivative kernel")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

def calc_local_wave_intervals(evts):
    wave_labels = evts.labels.astype(int)
    unique_labels = np.sort(np.unique(wave_labels))
    unique_channels = np.sort(np.unique(evts.array_annotations['channels'].astype(int)))

    channel_idx_map = np.empty(np.max(unique_channels)+1) * np.nan
    for i, channel in enumerate(unique_channels):
        channel_idx_map[channel] = i

    trigger_collection = np.empty((len(unique_labels), len(unique_channels)),
                                  dtype=float) * np.nan

    for (i, wave_id) in enumerate(unique_labels):
        wave_trigger_evts = evts[wave_labels == wave_id]

        channels = wave_trigger_evts.array_annotations['channels'].astype(int)

        channel_idx = channel_idx_map[channels].astype(int)
        trigger_collection[i, channel_idx] = wave_trigger_evts.times

    intervals = np.diff(trigger_collection, axis=0)
    intervals = intervals.reshape((len(unique_labels)-1)*len(unique_channels))

    mask = np.isfinite(intervals)
    intervals = intervals[mask]

    channel_ids = np.tile(unique_channels, len(unique_labels)-1)[mask]
    wave_ids = np.repeat(unique_labels[:-1], len(unique_channels))[mask]

    return wave_ids, channel_ids, intervals*evts.times.units



if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    imgseq = analogsignal_to_imagesequence(asig)

    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels != '-1']

    wave_ids, channel_ids, intervals = calc_local_wave_intervals(evts)

    # transform to DataFrame
    df = pd.DataFrame(intervals.magnitude, columns=['inter_wave_interval_local'])
    df['inter_wave_interval_local_unit'] = [intervals.dimensionality.string]*len(channel_ids)
    df['channel_id'] = channel_ids
    df[f'{args.event_name}_id'] = wave_ids

    df.to_csv(args.output)

    fig, ax = plt.subplots()
    ax.hist(1./intervals.magnitude[np.where(np.isfinite(1./intervals))[0]],
            bins=100, range=[0, 8])
    plt.xlabel('local rate of waves (Hz)', fontsize=7.)
    if args.output_img is not None:
        save_plot(args.output_img)
