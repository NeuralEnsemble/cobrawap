"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from utils import load_neo, save_plot, none_or_str
from scipy.ndimage import convolve as conv
from utils import AnalogSignal2ImageSequence
from apply_kernel import convolve_triggers

def calc_local_velocities(wave_evts, dim_x, dim_y, kernel_name):
    evts = wave_evts[wave_evts.labels != '-1']
    labels = evts.labels.astype(int)

    scale = evts.annotations['spatial_scale'].magnitude
    unit = evts.annotations['spatial_scale'].units / evts.times.units

    channel_ids = np.empty([dim_x, dim_y]) * np.nan
    channel_ids[evts.array_annotations['x_coords'].astype(int),
                evts.array_annotations['y_coords'].astype(int)] = evts.array_annotations['channels']
    channel_ids = channel_ids.reshape(-1)

    velocities = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channels = np.array([], dtype=int)

    for wave_id in np.unique(labels):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        convolved_trigger_X, convolved_trigger_Y = convolve_triggers(trigger_collection, kernel_name)
        t_x = convolved_trigger_X.reshape(-1)
        t_y = convolved_trigger_Y.reshape(-1)
    
        ## gradient based local velocity:
        v = scale * np.sqrt(1/(t_x**2 + t_y**2))

        channel_idx = np.where(np.isfinite(v))[0]

        velocities = np.append(velocities, v[channel_idx])
        channels = np.append(channels, channel_ids[channel_idx])
        wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channel_idx)))

    return wave_ids, channels, velocities*unit


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--kernel_name", nargs='?', type=str, default="Simple",
                     help="name of the kernel to be used")

    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)

    imgseq = block.segments[0].imagesequences[0]
    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]
    dim_t, dim_x, dim_y = np.shape(imgseq)
    wave_ids, channel_ids, velocities = calc_local_velocities(evts, dim_x, dim_y, args.kernel_name)


    # transform to DataFrame
    df = pd.DataFrame(list(zip(wave_ids, velocities.magnitude)),
                      columns=['wave_id', 'velocity_local'],
                      index=channel_ids)
    df['velocity_local_unit'] = [velocities.dimensionality.string]*len(channel_ids)
    df.index.name = 'channel_id'
    df.to_csv(args.output)

    plt.subplots()
    if args.output_img is not None:
        save_plot(args.output_img)
