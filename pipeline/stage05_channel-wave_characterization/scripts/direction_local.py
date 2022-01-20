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


def calc_local_directions(evts, kernel_name):
    labels = evts.labels.astype(int)
    dim_x = int(max(evts.array_annotations['x_coords']))+1
    dim_y = int(max(evts.array_annotations['y_coords']))+1

    scale = evts.annotations['spatial_scale'].magnitude
    unit = pq.radians

    directions = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channel_ids = np.array([], dtype=int)

    for wave_id in np.unique(labels):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)
        channels = wave_trigger_evts.array_annotations['channels'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        kernel = get_kernel(kernel_name)
        t_x = nan_conv2d(trigger_collection, kernel.x)[x_coords, y_coords]
        t_y = nan_conv2d(trigger_collection, kernel.y)[x_coords, y_coords]

        ## gradient based local directions:
        angle = np.arctan2(t_x, t_y)
        # angle = t_x + t_y*1j
        angle[~np.isfinite(angle)] = np.nan

        directions = np.append(directions, angle)
        channel_ids = np.append(channel_ids, channels)
        wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channels)))

    return wave_ids, channel_ids, directions*unit


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
    wave_ids, channel_ids, directions = calc_local_directions(evts, args.kernel)

    # transform to DataFrame
    df = pd.DataFrame(list(zip(wave_ids, directions.magnitude)),
                      columns=[f'{args.event_name}_id', 'direction_local'],
                      index=channel_ids)
    df['direction_local_unit'] = [directions.dimensionality.string]*len(channel_ids)
    df.index.name = 'channel_id'

    df.to_csv(args.output)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(directions.magnitude, bins=36, range=[-np.pi, np.pi])

    if args.output_img is not None:
        save_plot(args.output_img)
