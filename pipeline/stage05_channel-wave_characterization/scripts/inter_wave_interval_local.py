"""
"""

import argparse
import numpy as np
import quantities as pq
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from utils.io import load_neo, save_plot
from utils.parse import none_or_str
from utils.neo_utils import analogsignals_to_imagesequences


def calc_local_wave_intervals(evts):
    labels = evts.labels.astype(int)
    dim_x = int(max(evts.array_annotations['x_coords']))+1
    dim_y = int(max(evts.array_annotations['y_coords']))+1

    scale = evts.annotations['spatial_scale'].magnitude
    unit = evts.times.units

    intervals_collection = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channel_ids = np.array([], dtype=int)

    for (i, wave_id) in enumerate(np.unique(labels)):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)
        channels = wave_trigger_evts.array_annotations['channels'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        # if this is not the first wave
        if i:
            intervals = trigger_collection - trigger_collection_pre
            intervals = intervals[x_coords, y_coords]
            intervals[~np.isfinite(intervals)] = np.nan

            intervals_collection = np.append(intervals_collection, intervals)
            channel_ids = np.append(channel_ids, channels)
            wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channels)))

        trigger_collection_pre = trigger_collection.copy()

    return wave_ids, channel_ids, intervals_collection*unit


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--kernel", "--KERNEL", nargs='?', type=none_or_str, default=None,
                     help="derivative kernel")
    CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                     help="name of neo.Event to analyze (must contain waves)")
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    block = analogsignals_to_imagesequences(block)

    imgseq = block.segments[0].imagesequences[0]
    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels != '-1']

    dim_t, dim_x, dim_y = np.shape(imgseq)
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
