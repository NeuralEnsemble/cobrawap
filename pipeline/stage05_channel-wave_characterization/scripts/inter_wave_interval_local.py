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
from utils.neo import analogsignals_to_imagesequences


def calc_local_wave_intervals(wave_evts, dim_x, dim_y):
    evts = wave_evts[wave_evts.labels != '-1']
    labels = evts.labels.astype(int)

    scale = evts.annotations['spatial_scale'].magnitude
    unit = evts.times.units

    channel_ids = np.empty([dim_x, dim_y]) * np.nan
    channel_ids[evts.array_annotations['x_coords'].astype(int),
                evts.array_annotations['y_coords'].astype(int)] = evts.array_annotations['channels']
    channel_ids = channel_ids.reshape(-1)

    intervals_collection = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channels = np.array([], dtype=int)

    for (i, wave_id) in enumerate(np.unique(labels)):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        if i != 0: # if this is not the first wave
            ## local intervals:
            intervals = trigger_collection - trigger_collection_pre
            channel_idx = np.where(np.isfinite(intervals))[0]

            intervals_collection = np.append(intervals_collection,
                                             intervals[channel_idx])
            channels = np.append(channels, channel_ids[channel_idx])
            wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channel_idx)))

        trigger_collection_pre = trigger_collection.copy()

    return wave_ids, channels, intervals_collection*unit


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

    dim_t, dim_x, dim_y = np.shape(imgseq)
    wave_ids, channel_ids, intervals = calc_local_wave_intervals(evts,
                                                                 dim_x, dim_y)

    # transform to DataFrame
    df = pd.DataFrame(list(zip(wave_ids, intervals.magnitude)),
                      columns=[f'{args.event_name}_id',
                                'inter_wave_interval_local'],
                      index=channel_ids)
    df['inter_wave_interval_local_unit'] = [intervals.dimensionality.string]*len(channel_ids)
    df.index.name = 'channel_id'

    annotation_idx = [np.argmax(evts.array_annotations['channels'] == id)
                                                        for id in channel_ids]
    for key, value in evts.array_annotations.items():
        df[key] = value[annotation_idx]

    df.to_csv(args.output)

    fig, ax = plt.subplots()
    ax.hist(1./intervals.magnitude[np.where(np.isfinite(1./intervals))[0]],
            bins=100, range=[0, 8])
    plt.xlabel('local rate of waves (Hz)', fontsize=7.)
    if args.output_img is not None:
        save_plot(args.output_img)
