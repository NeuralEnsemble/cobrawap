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
    images = measure.block_reduce(images, (1, MACRO_PIXEL_DIM, MACRO_PIXEL_DIM), np.mean, cval = np.median(images))

    return images



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

    CLI.add_argument("--macro_pixel_dim",  nargs='?', type=int,
                      help="smoothing factor", default=2)

    
    args = CLI.parse_args()

    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[-1]
    
    MACRO_PIXEL_DIM = args.macro_pixel_dim    
    img_reduced = spatial_smoothing(imgseq, MACRO_PIXEL_DIM)

    if args.output_array is not None:
        np.save(args.output_array, img_reduced[0])
    if args.output_img is not None:
        plt.figure()
        plt.imshow(img_reduced[0])
        save_plot(args.output_img)

    dim_t = len(img_reduced[:,0,0])
    dim_x = len(img_reduced[0,:,0])
    dim_y = len(img_reduced[0,0,:])

    imgseq_reduced = neo.ImageSequence(img_reduced,
                                   units='dimensionless',
                                   spatial_scale=imgseq.spatial_scale * MACRO_PIXEL_DIM,
                                   sampling_rate=imgseq.sampling_rate,
                                   name='Reduced Images',
                                   description='Spatial downsampling',
                                   file_origin=imgseq.file_origin,
                                   **imgseq.annotations)

    block.segments[0].imagesequences = [imgseq_reduced]
    block = ImageSequence2AnalogSignal(block)

    asig = block.segments[0].analogsignals[-1]

    write_neo(args.output, block)
