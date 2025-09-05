"""
Downsample the input data by combining and averaging neighboring channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import neo
from skimage import measure
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_path
from utils.neo_utils import analogsignal_to_imagesequence, imagesequence_to_analogsignal

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path,
                 help="path of output image", default=None)
CLI.add_argument("--macro_pixel_dim", nargs='?', type=int,
                 help="smoothing factor", default=2)

def spatial_smoothing(imgseq, macro_pixel_dim):
    images_reduced = measure.block_reduce(imgseq.as_array(),
                                          block_size=(1, macro_pixel_dim, macro_pixel_dim),
                                          func=np.nanmean,
                                          cval=np.nan) #np.nanmedian(imgseq.as_array()))

    imgseq_reduced = neo.ImageSequence(images_reduced,
                                   units=imgseq.units,
                                   spatial_scale=imgseq.spatial_scale * macro_pixel_dim,
                                   macro_pixel_dim=macro_pixel_dim,
                                   sampling_rate=imgseq.sampling_rate,
                                   file_origin=imgseq.file_origin,
                                   t_start=imgseq.t_start)

    if 'array_annotations' in imgseq.annotations:
        del imgseq.annotations['array_annotations']

    imgseq_reduced.annotations.update(imgseq.annotations)

    if imgseq.name:
        imgseq_reduced.name = imgseq.name
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
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    imgseq = analogsignal_to_imagesequence(asig)

    imgseq_reduced = spatial_smoothing(imgseq, args.macro_pixel_dim)

    if args.output_img is not None:
        plot_downsampled_image(imgseq_reduced.as_array()[0], args.output_img)

    new_asig = imagesequence_to_analogsignal(imgseq_reduced)

    block.segments[0].analogsignals[0] = new_asig

    write_neo(args.output, block)
