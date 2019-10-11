import numpy as np
import argparse
import neo
import os


def normalize(images, normalize_by):
    if normalize_by == 'median':
        norm_function = np.median
    elif normalize_by == 'max':
        norm_function = np.max
    elif normalize_by == 'mean':
        norm_function = np.mean
    else:
        raise InputError("The method to normalize by is not recognized. \
                          Please choose either 'mean', 'median', or 'max'.")

    dim_t, dim_x, dim_y = images.shape
    coords = list(itertools.product(np.arange(dim_x), np.arange(dim_y)))
    norm_images = images.as_array()
    for x,y in coords:
        norm_value = norm_function(norm_images[:,x,y])
        if norm_value:
            norm_images[:,x,y] /= norm_value
        else:
            print('Normalization factor is {} for channel ({},{}) \
                   and was skipped.'.format(nom_value, x, y))
    for num in range(dim_t):
        images[num] = norm_images[num]
    del norm_images
    return images


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--normalize_by", nargs='?', type=str)
    args = CLI.parse_args()

    # load images
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    remove_annotations([block] + block.segments
                       + block.segments[0].analogsignals)

    images = normalize(block.segments[0].analogsignals[0], args.normalize_by)

    # save processed data
    images.name += ""
    images.description += "Normalized by {} ({})."\
                          .format(args.normalize_by, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = images

    with neo.NixIO(args.output) as io:
        io.write(block)
