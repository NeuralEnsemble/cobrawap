import numpy as np
import argparse
import neo
import quantities as pq
import matplotlib.pyplot as plt
import json
import os
import sys
from utils import parse_string2dict, ImageSequence2AnalogSignal
from utils import none_or_float, none_or_str, write_neo, time_slice


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data directory")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=None,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--orientation_top", nargs='?', type=str, required=True,
                     help="upward orientation of the recorded cortical region")
    CLI.add_argument("--orientation_right", nargs='?', type=str, required=True,
                     help="right-facing orientation of the recorded cortical region")
    args = CLI.parse_args()

    annotations = parse_string2dict(args.annotations)

    # local oscillation
    f = annotations['oscillation_freq'] * np.pi  # in Hz
    osciallation_function = lambda t: np.sin(f*t)

    # spatial propagation
    pixel_shift = args.spatial_scale/annotations['velocity']

    # propagation direction
    direction_vec = np.exp(1j*annotations['direction'])
    dx, dy = np.real(direction_vec), np.imag(direction_vec)

    # build signal array
    Nx, Ny = annotations['grid_size']
    t = np.linspace(0, int(args.t_stop), int(args.t_stop*args.sampling_rate))
    signal = np.zeros((len(t), Ny, Nx))
    for col in range(Nx):
        for row in range(Ny):
            signal[:,row,col] = osciallation_function(t - pixel_shift*(col*dx + row*dy))

    imgseq = neo.ImageSequence(signal, units='dimensionless',
                               sampling_rate=args.sampling_rate*pq.Hz,
                               spatial_scale=args.spatial_scale*pq.mm)

    block = neo.Block()
    seg = neo.Segment()
    block.segments.append(seg)
    block.segments[0].imagesequences.append(imgseq)
    block = ImageSequence2AnalogSignal(block)

    if args.annotations is not None:
        block.segments[0].analogsignals[0].annotations.update(annotations)

    block.segments[0].analogsignals[0].annotations.update(orientation_top=args.orientation_top)
    block.segments[0].analogsignals[0].annotations.update(orientation_right=args.orientation_right)

    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'artificially generate data (neo version {}). '\
                                    .format(neo.__version__)
    block.segments[0].analogsignals[0].description = ''

    # Save data
    write_neo(args.output, block)
