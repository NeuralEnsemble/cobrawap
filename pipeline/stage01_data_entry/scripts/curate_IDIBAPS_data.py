import numpy as np
import argparse
import quantities as pq
import re
import os
import sys
import neo
from utils import load_neo, write_neo, time_slice, flip_image, rotate_image
from utils import AnalogSignal2ImageSequence, ImageSequence2AnalogSignal
from utils import parse_string2dict, none_or_float, none_or_str

def merge_analogsingals(asigs):
    # ToDo: to be replaced by neo utils functions
    min_length = np.min([len(asig.times) for asig in asigs])
    max_length = np.max([len(asig.times) for asig in asigs])
    if min_length != max_length:
        print('Warning: the length of the analog signals differs '\
            + 'between {} and {} '.format(min_length, max_length)\
            + 'All signals will be cut to the same length and merged '\
            + 'into one AnalogSignal object.')

    if len(np.unique([asig.sampling_rate for asig in asigs])) > 1:
        raise ValueError('The AnalogSignal objects have different '\
                       + 'sampling rates!')

    asig_array = np.zeros((min_length, len(asigs)))

    for channel_number, asig in enumerate(asigs):
        asig_array[:, channel_number] = np.squeeze(asig.as_array()[:min_length])

    merged_asig = neo.AnalogSignal(asig_array*asigs[0].units,
                                   sampling_rate=asigs[0].sampling_rate,
                                   t_start=asigs[0].t_start)
    for key in asigs[0].annotations.keys():
        try:
            merged_asig.array_annotations[key] = np.array([a.annotations[key]
                                                           for a in asigs])
        except:
            print('Can not merge annotation ', key)
    return merged_asig


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=None,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--orientation_top", nargs='?', type=str, required=True,
                     help="upward orientation of the recorded cortical region")
    CLI.add_argument("--orientation_right", nargs='?', type=str, required=True,
                     help="right-facing orientation of the recorded cortical region")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    args = CLI.parse_args()

    try:
        block = load_neo(args.data, try_signal_grouping=True)
    except Exception as e:
        print(e)
        block = load_neo(args.data, try_signal_grouping=False)

    asigs = block.segments[0].analogsignals

    if len(asigs) > 1:
        print(f'Merging {len(asigs)} AnalogSignals into one.')
        asig = merge_analogsingals(asigs)
    else:
        asig = asigs[0]

    asig = time_slice(asig, args.t_start, args.t_stop)

    # add metadata
    kwargs = parse_string2dict(args.kwargs)

    channels = asig.array_annotations[kwargs['ELECTRODE_ANNOTATION_NAME']]

    coords = np.array([kwargs['NAME2COORDS'][str(channel)] for channel in channels.astype(int)])
    asig.array_annotations.update(x_coords=coords[:,0])
    asig.array_annotations.update(y_coords=coords[:,1])

    # locations = []
    # for channel in channels:
    #     locations.append([loc for loc in kwargs['ELECTRODE_LOCATION'].keys()
    #                       if channel in kwargs['ELECTRODE_LOCATION'][loc]][0])
    # asig.array_annotations.update(electrode_location=locations)

    # colors = [kwargs['ELECTRODE_COLOR'][loc] for loc in
    #           asig.array_annotations['electrode_location']]
    # asig.array_annotations.update(electrode_color=colors)

    asig.annotations.update(parse_string2dict(args.annotations))
    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)
    asig.annotations.update(orientation_top=args.orientation_top)
    asig.annotations.update(orientation_right=args.orientation_right)

    if asig.description is None:
        asig.description = ''
    asig.description += 'ECoG signal. '

    # dim_t, channel_num = asig.as_array().shape
    # chidx = neo.ChannelIndex(name=asig.name,
    #                          channel_ids=np.arange(channel_num),
    #                          index=np.arange(channel_num),
    #                          coordinates=coords*args.spatial_scale*pq.mm)
    # chidx.annotations.update(asig.array_annotations)

    block.segments[0].analogsignals = [asig]

    # change data orientation to be top=ventral, right=lateral
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[0]
    imgseq = flip_image(imgseq, axis=-2)
    block.segments[0].imagesequences[0] = imgseq
    block.segments[0].analogsignals.clear()
    block = ImageSequence2AnalogSignal(block)

    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.Spike2IO (neo version {})'\
                                    .format(neo.__version__)



    # block.channel_indexes.append(chidx)
    # block.segments[0].analogsignals[0].channel_index = chidx
    # chidx.analogsignals.append(asig)
    block.channel_indexes = []

    write_neo(args.output, block)
