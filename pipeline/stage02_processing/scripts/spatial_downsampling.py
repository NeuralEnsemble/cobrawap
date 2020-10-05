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


def spatial_smoothing(images, macro_pixel_dim):

    # Now we need to reduce the noise from the images by performing a spatial smoothing
    images_reduced = measure.block_reduce(images, (1, macro_pixel_dim, macro_pixel_dim), np.nanmean, cval = np.nanmedian(images))

    dim_t, dim_x, dim_y = images_reduced.shape
    imgseq_reduced = neo.ImageSequence(images_reduced,
                                   units=images.units,
                                   spatial_scale=images.spatial_scale * macro_pixel_dim,
                                   sampling_rate=images.sampling_rate,
                                   file_origin=images.file_origin,
                                   **imgseq.annotations)

    imgseq_reduced.name = images.name + " "
    imgseq_reduced.annotations.update(macro_pixel_dim=macro_pixel_dim)
    imgseq_reduced.description = images.description +  "spatially downsampled ({})." .format(os.path.basename(__file__))

    return imgseq_reduced

def plot_downsampled_image(image, output_path):
    plt.figure()
    plt.imshow(image, interpolation='nearest', cmap='viridis', origin='lower')
    save_plot(output_path)

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
                      help="smoothing factor", default=2)
    
    args = CLI.parse_args()
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[0]
    
    imgseq_reduced = spatial_smoothing(imgseq, args.macro_pixel_dim)

    if args.output_img is not None:
        plot_downsampled_image(images_reduced.as_array()[0], args.output_img)

    block.segments[0].imagesequences = [imgseq_reduced]
    block.segments[0].analogsignals.clear()
    block = ImageSequence2AnalogSignal(block)

    write_neo(args.output, block)
