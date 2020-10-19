"""
Substract the background of a given dataset by substracting the mean of each channel.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import load_neo, write_neo, save_plot, none_or_str

def shape_frame(value_array, coords):
    dim_x = np.max(coords[:,0]) + 1
    dim_y = np.max(coords[:,1]) + 1
    frame = np.empty((dim_x, dim_y)) * np.nan
    for pixel, xy in zip(value_array, coords):
        frame[int(xy[0]), int(xy[1])] = pixel
    return frame

def plot_frame(frame):
    fig, ax = plt.subplots()
    ax.imshow(frame, interpolation='nearest', cmap=plt.cm.gray, origin='lower')
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)
    CLI.add_argument("--output_array",  nargs='?', type=none_or_str,
                      help="path of output numpy array", default=None)
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    signal = asig.as_array()
    background = np.nanmean(signal, axis=0)
    signal -= background

    if args.output_img or args.output_array is not None:
        coords = np.array([(x,y) for x,y in
                           zip(asig.array_annotations['x_coords'],
                               asig.array_annotations['y_coords'])],
                          dtype=int)
        frame = shape_frame(background, coords)
        if args.output_array is not None:
            np.save(args.output_array, frame)
        if args.output_img is not None:
            plot_frame(frame)
            save_plot(args.output_img)

    new_asig = asig.duplicate_with_new_data(signal)
    new_asig.array_annotations = asig.array_annotations
    new_asig.name += ""
    new_asig.description += "The mean of each channel was substracted ({})."\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = new_asig

    write_neo(args.output, block)
