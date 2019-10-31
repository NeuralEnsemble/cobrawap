"""
Substracts the background of a given dataset by substracting the mean of each
channel.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import neo
import os
import sys
import re
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape, remove_annotations


def substract_background(asig, background):
    for num, frame in enumerate(asig):
        asig[num] = frame - background
    return asig

def determine_spatial_scale(coords):
    coords = np.array(coords)
    dists = np.diff(coords[:,0])
    dists = dists[np.nonzero(dists)]
    return np.min(dists)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--output_img", nargs='?', type=str)
    CLI.add_argument("--output_array", nargs='?', type=str)
    args = CLI.parse_args()

    # load data
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    # remove_annotations([block] + block.segments
    #                    + block.segments[0].analogsignals)

    asig = block.segments[0].analogsignals[0]

    # calculate average signal per channel as backgound
    background = np.mean(asig, axis=0)

    asig = substract_background(asig, background)

    # save background as numpy array
    chidx = block.channel_indexes[0]
    spatial_scale = determine_spatial_scale(chidx.coordinates)\
                  * chidx.coordinates[0][0].units
    dim_x = np.max(np.array(chidx.coordinates)[:,0])/spatial_scale.magnitude
    dim_y = np.max(np.array(chidx.coordinates)[:,1])/spatial_scale.magnitude
    bkgr_img = np.empty((int(round(dim_x)), int(round(dim_y)))) * np.nan
    for pixel, coords in zip(background, chidx.coordinates):
        bkgr_img[int(coords[0])][int(coords[1])] = pixel

    np.save(args.output_array, bkgr_img)

    # save background image
    fig, ax = plt.subplots()
    ax.imshow(bkgr_img, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.output_img)

    # save processed data
    asig.name += ""
    asig.description += "The mean of each channel was substracted ({})."\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
