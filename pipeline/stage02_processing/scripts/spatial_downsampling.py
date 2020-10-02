"""
Spatial downsampling of the input dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import neo
import quantities as pq
from skimage import data, io, filters, measure
from utils import determine_spatial_scale, load_neo, write_neo, save_plot, \
                  none_or_str, AnalogSignal2ImageSequence, ImageSequence2AnalogSignal


def spatial_smoothing(images, MACRO_PIXEL_DIM):

    # Now we need to reduce the noise from the images by performing a spatial smoothing
    images_reduced = measure.block_reduce(images, (1, MACRO_PIXEL_DIM, MACRO_PIXEL_DIM), np.mean, cval = np.median(images))

    if args.output_img is not None:
        plt.figure()
        plt.imshow(images_reduced[0], interpolation='nearest', cmap='viridis', origin='lower')
        save_plot(args.output_img)


    dim_t, dim_x, dim_y = images_reduced.shape
    imgseq_reduced = neo.ImageSequence(images_reduced,
                                   units=images.units,
                                   spatial_scale=images.spatial_scale * MACRO_PIXEL_DIM,
                                   sampling_rate=images.sampling_rate,
                                   name='Reduced Images',
                                   description='Spatial downsampling',
                                   file_origin=images.file_origin,
                                   **imgseq.annotations)

    imgseq_reduced.name += " "
    imgseq_reduced.annotations.update(macro_pixel_dim=MACRO_PIXEL_DIM)
    imgseq_reduced.description += "spatially downsampled ({})." .format(os.path.basename(__file__))

    return imgseq_reduced



if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)

    CLI.add_argument("--macro_pixel_dim",  nargs='?', type=int,
                      help="smoothing factor", default=4)

    
    args = CLI.parse_args()

    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[0]
    
    MACRO_PIXEL_DIM = args.macro_pixel_dim
    print('Macro', MACRO_PIXEL_DIM)
    imgseq_reduced = spatial_smoothing(imgseq, MACRO_PIXEL_DIM)
 

    block.segments[0].imagesequences = [imgseq_reduced]
    block.segments[0].analogsignals.clear()
    block = ImageSequence2AnalogSignal(block)

    write_neo(args.output, block)
