import numpy as np
import argparse
import quantities as pq
import re
import os
import sys
import neo
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import parse_string2dict, check_analogsignal_shape

def merge_analogsingals(asigs):
    min_length = np.min([len(asig.times) for asig in asigs])
    max_length = np.max([len(asig.times) for asig in asigs])
    if min_length != max_length:
        print('Warning: the length of the analog signals differs '\
            + 'between {} and {} '.format(min_length, max_length)\
            + 'All signals will be cut to the same length and merged '\
            + 'into one AnalogSignal object.')

    if len(np.unique([asig.sampling_rate for asig in asigs])) > 1:
        print([asig.sampling_rate for asig in asigs])
        raise ValueError('The AnalogSignal objects have different '\
                       + 'sampling rates!')

    asig_array = np.zeros((min_length, len(asigs)))

    for channel_number, asig in enumerate(asigs):
        asig_array[:, channel_number] = np.squeeze(asig.as_array())

    # ToDo: check if annotations have same keys
    array_annotations = {}
    for key in asigs[0].annotations.keys():
        array_annotations[key] = []
        for asig in asigs:
            array_annotations[key] += [asig.annotations[key]]

    array_annotations['name'] = []
    for asig in asigs:
        array_annotations['name'] += asig.name

    return neo.AnalogSignal(asig_array,
                                sampling_rate=asigs[0].sampling_rate,
                                t_start=asigs[0].t_start,
                                t_stop=asigs[0].t_stop,
                                array_annotations=array_annotations)

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

    asigs = block.segments[0].analogsignals

    if len(asigs) > 1:
        print('Merging {} AnalogSignals into one.'.format(len(asigs)))
        asig = merge_analogsingals(asigs)
    else:
        asig = asigs[0]

    # ToDo: In case of multiple AnalogSignal objects with slightly different
    #       length. Cut and merge them into one AnalogSignal.
    # ToDo: Move fomat check to a separate validation block
    check_analogsignal_shape(block.segments[0].analogsignals)

    # add metadata
    kwargs = parse_string2dict(args.kwargs)

    channels = asig.array_annotations[kwargs['ELECTRODE_ANNOTATION_NAME']]

    coords = np.array([kwargs['NAME2COORDS'][str(channel)] for channel in channels])
    asig.array_annotations.update(x_coords=coords[:,0])
    asig.array_annotations.update(y_coords=coords[:,1])

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

    dim_t, channel_num = asig.as_array().shape

    chidx = neo.ChannelIndex(name=asig.name,
                             channel_ids=np.arange(channel_num),
                             index=np.arange(channel_num),
                             coordinates=coords*args.spatial_scale*pq.mm)
    chidx.annotations.update(asig.array_annotations)

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
    block.channel_indexes.append(chidx)
    block.segments[0].analogsignals[0].channel_index = chidx
    chidx.analogsignals.append(asig)

    with neo.NixIO(args.output) as io:
        io.write(block)
