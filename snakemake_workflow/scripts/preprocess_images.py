import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import shapely.geometry as geo
import argparse
import neo
import os
import itertools


def substract_background(images, background=None):
    for num, frame in enumerate(images):
        images[num] = frame - background
    return images


def apply_mask(images, mask):
    for num, frame in enumerate(images):
        frame[np.bitwise_not(mask)] = np.nan
        images[num] = frame
    return images


def spatial_smoothing(images, macro_pixel_dim):
    smoothed_images = sk.measure.block_reduce(images,
                               (1, macro_pixel_dim, macro_pixel_dim),
                               np.mean)

    smoothed_asig = neo.AnalogSignal(smoothed_images,
                            units=images.units,
                            sampling_rate=images.sampling_rate,
                            t_start=images.t_start,
                            t_stop=images.t_stop,
                            file_origin=images.file_origin,
                            pixel_size=images.annotations['pixel_size']
                                       *macro_pixel_dim)
    return smoothed_asig


def normalize(images, normalize_by):
    if normalize_by == 'median':
        norm_function = np.mean
    elif normalize_by == 'max':
        norm_function = np.max
    else:
        raise InputError("The method to normalize by is not recognized. Please choose either 'median', or 'max'.")

    dim_t, dim_x, dim_y = images.shape
    coords = list(itertools.product(np.arange(dim_x), np.arange(dim_y)))
    norm_images = images.as_array()
    for x,y in coords:
        norm_images[:,x,y] /= norm_function(norm_images[:,x,y])
    for num in range(dim_t):
        images[num] = norm_images[num]
    return images


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def none_or_str(value):
    if value == 'None':
        return None
    return str(value)

if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--background", nargs='?', type=none_or_str)
    CLI.add_argument("--mask", nargs='?', type=none_or_str)
    CLI.add_argument("--macro_pixel_dim", nargs='?', type=none_or_int)
    CLI.add_argument("--normalize_by", nargs='?', type=none_or_str)
    CLI.add_argument("--output", nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        images = io.read_block().segments[0].analogsignals[0]

    if args.background is not None:
        images = substract_background(images, np.load(args.background))

    if args.normalize_by is not None and args.normalize_by != 'None':
        images = normalize(images, args.normalize_by)
    else:
        args.normlize_by = None

    if args.mask is not None:
        images = apply_mask(images, np.load(args.mask))

    if args.macro_pixel_dim is not None and args.macro_pixel_dim != 'None':
        images = spatial_smoothing(images, args.macro_pixel_dim)
    else:
        args.macro_pixel_dim = None

    # Save as NIX file
    description = '{} {} {} {}'.format('' if args.background is None
                                        else 'background substracted;',
                                        '' if args.normalize_by is None
                                        else 'normlized by '
                                        + args.normalize_by,
                                        '' if args.mask is None
                                        else 'mask applied;',
                                        '' if args.macro_pixel_dim is None
                                        else 'resolution reduced by factor '
                                        + str(args.macro_pixel_dim))

    image_block = neo.Block(name='Results of {}'\
                                 .format(os.path.basename(__file__)))
    seg = neo.Segment(name='Segment 1', description=description)
    image_block.segments.append(seg)
    image_block.segments[0].analogsignals.append(images)
    with neo.NixIO(args.output) as io:
        io.write(image_block)
