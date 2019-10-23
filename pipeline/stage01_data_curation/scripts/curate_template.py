import numpy as np
import argparse
import neo
import quantities as pq
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import parse_string2dict, check_analogsignal_shape, ImageSequence2AnalogSignal

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def none_or_str(value):
    if value == 'None':
        return None
    return str(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float)
    CLI.add_argument("--spatial_scale", nargs='?', type=float)
    CLI.add_argument("--data_name", nargs='?', type=str)
    CLI.add_argument("--annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str)
    args = CLI.parse_args()

    # Load data with Neo IO or custom loading routine
    io = neo.Spike2IO(args.data)
    block = io.read_block()
    # If there is no Neo IO for the data type available,
    # the data must be loaded conventioally and added to a newly constructed
    # Neo block. For building a Neo objects, have a look into the documentation
    # https://neo.readthedocs.io/

    # In case the dataset is imagaging data and therefore stored as an
    # ImageSequence object, it needs to be transformed into an AnalogSignal
    # object. To do this use the function ImageSequence2AnalogSignal in utils.py

    # check that there is only one AnalogSignal object
    # with shape (# time steps, # channels)
    check_analogsignal_shape(block.segments[0].analogsignals)

    asig = block.segments[0].analogsignals[0]

    # Add metadata from ANNOTIATION dict
    asig.annotations.update(parse_string2dict(args.annotations))
    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)

    # Add metadata from ARRAY_ANNOTIATION dict
    asig.array_annotations.update(parse_string2dict(args.array_annotations))

    # Do custom metadata processing from KWARGS dict (optional)
    # kwargs = parse_string2dict(args.kwargs)
    # ... do something

    # Add description to the Neo object
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.Spike2IO (neo version {})'\
                                    .format(neo.__version__)
    if asig.description is None:
        asig.description = ''
    asig.description += 'ECoG signal. '

    # Update the annotated AnalogSignal object in the Neo Block
    block.segments[0].analogsignals[0] = asig

    # Save data into Nix file
    with neo.NixIO(args.output) as io:
        io.write(block)
