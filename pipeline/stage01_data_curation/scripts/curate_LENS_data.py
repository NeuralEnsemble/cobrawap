import numpy as np
import argparse
import neo
import quantities as pq
import json
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import parse_string2dict, ImageSequence2AnalogSignal

def none_or_str(value):
    if value == 'None':
        return None
    return str(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--sampling_rate", nargs='?', type=float)
    CLI.add_argument("--spatial_scale", nargs='?', type=float)
    CLI.add_argument("--t_start", nargs='?', type=float)
    CLI.add_argument("--t_stop", nargs='?', type=float)
    CLI.add_argument("--data_name", nargs='?', type=str)
    CLI.add_argument("--annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str)
    args = CLI.parse_args()

    # Load optical data
    io = neo.io.tiffio.TiffIO(directory_path=args.data,
                              sampling_rate=args.sampling_rate*pq.Hz,
                              spatial_scale=args.spatial_scale*pq.mm,
                              units='dimensionless')

    block = io.read_block()

    # Transform into analogsignals
    block = ImageSequence2AnalogSignal(block)
    if len(block.segments[0].analogsignals) > 1:
        raise IOError("Additional analog signals detected! "\
                    + "This pipeline only operates on single AnalogSignals.")

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
    with neo.NixIO(args.output) as io:
        io.write(block)
