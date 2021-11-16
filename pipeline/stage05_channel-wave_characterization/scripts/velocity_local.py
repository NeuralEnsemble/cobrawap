"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_neo, save_plot, none_or_str
from scipy.ndimage import convolve as conv
from utils import AnalogSignal2ImageSequence


def calc_local_velocities(wave_evts, dim_x, dim_y):
    evts = wave_evts[wave_evts.labels != '-1']
    labels = evts.labels.astype(int)
    print(labels)
    x_coords = evts.array_annotations['x_coords'].astype(int)
    y_coords = evts.array_annotations['y_coords'].astype(int)
    scale = evts.annotations['spatial_scale'].magnitude
    unit = evts.annotations['spatial_scale'].units / evts.times.units

    wave_collection = np.empty([len(labels), dim_x, dim_y]) * np.nan
    wave_collection[labels, x_coords, y_coords] = evts.times

    # channel_ids = np.empty([len(labels), dim_x, dim_y]) * np.nan
    # channel_ids[labels, x_coords, y_coords] = evts.array_annotations['channels']

    # ToDo: use derivate kernel convolution instead (while ignoring nans)
    Tx = np.diff(wave_collection, axis=1, append=np.nan) #[:, :dim_x-1, :dim_y-1]
    Ty = np.diff(wave_collection, axis=2, append=np.nan) #[:, :dim_x-1, :dim_y-1]
    # channel_ids = channel_ids[:, :dim_x-1, :dim_y-1]

    Tx = np.reshape(Tx, (len(labels), -1))
    Ty = np.reshape(Ty, (len(labels), -1))
    # channel_ids = np.reshape(channel_ids, (len(labels), -1)).flatten()

    velocities = np.sqrt(2*scale**2/(Tx**2 + Ty**2))
    wave_ids, channel_idx = np.where(np.isfinite(velocities))

    velocities = velocities[wave_ids, channel_idx]
    # print(channel_idx == channel_ids[channel_idx])

    return wave_ids, channel_idx, velocities * unit


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")

    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)

    imgseq = block.segments[0].imagesequences[0]
    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]

    dim_t, dim_x, dim_y = np.shape(imgseq)
    wave_ids, channel_ids, velocities = calc_local_velocities(evts, dim_x, dim_y)


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
