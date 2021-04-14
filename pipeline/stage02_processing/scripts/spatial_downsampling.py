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


def spatial_smoothing(imgseq, macro_pixel_dim):
    images_reduced = measure.block_reduce(imgseq.as_array(),
                                          block_size=(1, macro_pixel_dim, macro_pixel_dim),
                                          func=np.nanmean,
                                          cval=np.nan) #np.nanmedian(imgseq.as_array()))

    dim_t, dim_x, dim_y = images_reduced.shape
    imgseq_reduced = neo.ImageSequence(images_reduced,
                                   units=imgseq.units,
                                   spatial_scale=imgseq.spatial_scale * macro_pixel_dim,
                                   sampling_rate=imgseq.sampling_rate,
                                   file_origin=imgseq.file_origin,
                                   t_start=imgseq.t_start)

    if 'array_annotations' in imgseq.annotations:
        del imgseq.annotations['array_annotations']

    imgseq_reduced.annotations.update(imgseq.annotations)

    imgseq_reduced.name = imgseq.name + " "
    imgseq_reduced.annotations.update(macro_pixel_dim=macro_pixel_dim)
    imgseq_reduced.description = imgseq.description +  \
                "spatially downsampled ({}).".format(os.path.basename(__file__))

    return imgseq_reduced

def plot_downsampled_image(image, output_path):
    plt.figure()
    plt.imshow(image, interpolation='nearest', cmap='viridis', origin='lower')
    save_plot(output_path)
    return plt.gca()

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
        plot_downsampled_image(imgseq_reduced.as_array()[0], args.output_img)

    new_block = neo.Block()
    new_segment = neo.Segment()
    new_block.segments.append(new_segment)
    new_block.segments[0].imagesequences.append(imgseq_reduced)
    new_block = ImageSequence2AnalogSignal(new_block)

    block.segments[0].analogsignals[0] = new_block.segments[0].analogsignals[0]

    write_neo(args.output, block)
