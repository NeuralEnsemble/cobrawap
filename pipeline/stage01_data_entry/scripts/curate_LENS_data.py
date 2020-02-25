import numpy as np
import argparse
import neo
import quantities as pq
import json
import os
import sys
from utils import parse_string2dict, ImageSequence2AnalogSignal,
                  none_or_float, none_or_str, write_neo


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
    args = CLI.parse_args()

    # Load optical data
    io = neo.io.tiffio.TiffIO(directory_path=args.data,
                              sampling_rate=args.sampling_rate*pq.Hz,
                              spatial_scale=args.spatial_scale*pq.mm,
                              units='dimensionless')

    block = io.read_block()

    # Transform into analogsignals
    block.segments[0].analogsignals = []
    block = ImageSequence2AnalogSignal(block)

    if args.annotations is not None:
        block.segments[0].analogsignals[0].annotations.\
                                    update(parse_string2dict(args.annotations))

    # ToDo: add metadata
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.TiffIO (neo version {}). '\
                                    .format(neo.__version__)
    if block.segments[0].analogsignals[0].description is None:
        block.segments[0].analogsignals[0].description = ''
    block.segments[0].analogsignals[0].description += 'Ca+ imaging signal. '

    # Save data
    write_neo(args.output, block)
