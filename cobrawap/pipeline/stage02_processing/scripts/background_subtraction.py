"""
Subtract the background of the input data by subtracting the mean of each channel.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path,
                 help="path of output image", default=None)
CLI.add_argument("--output_array", nargs='?', type=none_or_path,
                 help="path of output numpy array", default=None)

def shape_frame(value_array, xy_coords):
    dim_x = np.max(xy_coords[:,0]) + 1
    dim_y = np.max(xy_coords[:,1]) + 1
    frame = np.empty((dim_y, dim_x)) * np.nan
    for pixel, (x,y) in zip(value_array, xy_coords):
        frame[int(y), int(x)] = pixel
    return frame

def plot_frame(frame):
    fig, ax = plt.subplots()
    ax.imshow(frame, interpolation='nearest', cmap=plt.cm.gray, origin='lower')
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    signal = asig.as_array()
    background = np.nanmean(signal, axis=0)
    signal -= background

    if args.output_img or args.output_array is not None:
        xy_coords = np.array([(x,y) for x,y in
                           zip(asig.array_annotations['x_coords'],
                               asig.array_annotations['y_coords'])],
                           dtype=int)
        frame = shape_frame(background, xy_coords)
        if args.output_array is not None:
            np.save(args.output_array, frame)
        if args.output_img is not None:
            plot_frame(frame)
            save_plot(args.output_img)

    new_asig = asig.duplicate_with_new_data(signal)
    new_asig.array_annotations = asig.array_annotations
    new_asig.description += "The mean of each channel was subtracted ({})."\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = new_asig

    write_neo(args.output, block)
