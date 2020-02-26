import numpy as np
import argparse
import quantities as pq
import re
import os
import sys
import neo
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
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    # slice
    asig = block.segments[0].analogsignals[0].time_slice(1*pq.s, 11*pq.s)

    # subsample
    subs = 10
    asig = neo.AnalogSignal(signal=asig.as_array()[::subs, :],
                           units=asig.units,
                           sampling_rate=asig.sampling_rate/subs,
                           name=asig.name,
                           description=asig.description,
                           file_origin=asig.file_origin,
                           array_annotations=asig.array_annotations,
                           **asig.annotations)

    # add metadata
    kwargs = parse_string2dict(args.kwargs)

    channels = asig.array_annotations[kwargs['ELECTRODE_ANNOTATION_NAME']]

    coords = np.array([kwargs['NAME2COORDS'][str(channel)] for channel in channels])
    asig.array_annotations.update(x_coords=coords[:,0])
    asig.array_annotations.update(y_coords=coords[:,1])

    asig.annotations.update(parse_string2dict(args.annotations))
    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)

    # Save data
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.NixIO (neo version {})'\
                                    .format(neo.__version__)
    if asig.description is None:
        asig.description = ''
    asig.description += 'Raw downsampled 10k Hz signal. '

    # Save data
    if len(block.segments[0].analogsignals) > 1:
        raise Warning('Additional AnalogSignal found. The pipeline can yet \
                       only process single AnalogSignals.')

    block.segments[0].analogsignals[0] = asig
    # block.channel_indexes.append(chidx)
    # block.segments[0].analogsignals[0].channel_index = chidx
    # chidx.analogsignals.append(asig)

    with neo.NixIO(args.output) as io:
        io.write(block)
