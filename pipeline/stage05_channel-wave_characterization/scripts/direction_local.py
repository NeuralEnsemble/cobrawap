"""
Compute local directions
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
from utils.convolve import nan_conv2d, get_kernel


def calc_local_directions(wave_evts, kernel_name):
    evts = wave_evts[wave_evts.labels != '-1']
    labels = evts.labels.astype(int)
    dim_x = max(evts.array_annotations['x_coords'])+1
    dim_y = max(evts.array_annotations['y_coords'])+1

    scale = evts.annotations['spatial_scale'].magnitude
    unit = pq.radians

    channel_ids = np.empty([dim_x, dim_y]) * np.nan
    channel_ids[evts.array_annotations['x_coords'].astype(int),
                evts.array_annotations['y_coords'].astype(int)] = evts.array_annotations['channels']
    channel_ids = channel_ids.reshape(-1)

    directions = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channels = np.array([], dtype=int)

    for wave_id in np.unique(labels):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        kernel = get_kernel(kernel_name)
        t_x = nan_conv2d(trigger_collection, kernel.x).reshape(-1)
        t_y = nan_conv2d(trigger_collection, kernel.y).reshape(-1)

        ## gradient based local directions:
        angle = np.arctan2(t_x, t_y)

        channel_idx = np.where(np.isfinite(angle))[0]

        directions = np.append(directions, angle[channel_idx])
        channels = np.append(channels, channel_ids[channel_idx])
        wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channel_idx)))

    return wave_ids, channels, directions*unit


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
    wave_ids, channel_ids, directions = calc_local_directions(evts, args.kernel)

    # transform to DataFrame
    df = pd.DataFrame(list(zip(wave_ids, directions.magnitude)),
                      columns=[f'{args.event_name}_id', 'directions_local'],
                      index=channel_ids)
    df['directions_local_unit'] = [directions.dimensionality.string]*len(channel_ids)
    df.index.name = 'channel_id'
    df.to_csv(args.output)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(directions.magnitude, bins=36, range=[-np.pi, np.pi])

    if args.output_img is not None:
        save_plot(args.output_img)
