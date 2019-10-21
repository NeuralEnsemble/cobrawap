import numpy as np
import argparse
import neo
import quantities as pq
import re
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import parse_string2dict, check_analogsignal_shape


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

    # Load data
    io = neo.Spike2IO(args.data)
    block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    asig = block.segments[0].analogsignals[0]

    # add metadata
    kwargs = parse_string2dict(args.kwargs)

    channels = asig.array_annotations[kwargs['ELECTRODE_ANNOTATION_NAME']]

    coords = [str(kwargs['NAME2COORDS'][str(channel)]) for channel in channels]
    asig.array_annotations.update(coords=coords)

    locations = []
    for channel in channels:
        locations.append([loc for loc in kwargs['ELECTRODE_LOCATION'].keys()
                          if channel in kwargs['ELECTRODE_LOCATION'][loc]][0])
    asig.array_annotations.update(electrode_location=locations)

    colors = [kwargs['ELECTRODE_COLOR'][loc] for loc in
              asig.array_annotations['electrode_location']]
    asig.array_annotations.update(electrode_color=colors)

    asig.annotations.update(parse_string2dict(args.annotations))
    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)

    # Save data
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.Spike2IO (neo version {})'\
                                    .format(neo.__version__)
    if asig.description is None:
        asig.description = ''
    asig.description += 'ECoG signal. '

    # Save data
    if len(block.segments[0].analogsignals) > 1:
        raise Warning('Additional AnalogSignal found. The pipeline can yet \
                       only process single AnalogSignals.')

    block.segments[0].analogsignals[0] = asig
    with neo.NixIO(args.output) as io:
        io.write(block)
