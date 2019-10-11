"""
Substracts the background of a given dataset by substracting the mean of each
channel.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import neo
import sys
sys.path.append('../../')
from utils import check_analogsignal_shape, remove_annotations


def substract_background(images, background):
    for num, frame in enumerate(images):
        images[num] = frame - background
    return images


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--output_img", nargs='?', type=str)
    CLI.add_argument("--output_array", nargs='?', type=str)
    args = CLI.parse_args()

    # load images
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    remove_annotations([block] + block.segments
                       + block.segments[0].analogsignals)

    images = block.segments[0].analogsignals[0]

    # calculate average image as backgound
    background = np.mean(images, axis=0)

    images = substract_background(images, background)

    # save background as numpy array
    np.save(args.output_array, background)

    # save background image
    fig, ax = plt.subplots()
    ax.imshow(background, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.output_img)

    # save processed data
    # ToDo: overwrite AnalogSignals or create new segement?
    images.name += ""
    images.description += "The mean of each channel was substracted ({})."\
                          .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = images

    with neo.NixIO(args.output) as io:
        io.write(block)
