import numpy as np
import argparse
import neo
from neo import utils as neo_utils
import nixio
import quantities as pq
import matplotlib.pyplot as plt
import json
import os
import sys
from utils import parse_string2dict, ImageSequence2AnalogSignal
from utils import none_or_float, none_or_str, none_or_int, write_neo, time_slice


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
    CLI.add_argument("--trial", nargs='?', type=none_or_int, default=None,
                     help="Trial number (optional), if None select all")
    args, unknown = CLI.parse_known_args()

    # Load data
    with neo.NixIO(args.data, 'ro') as io:
        blocks = io.read_all_blocks()

    annotations = parse_string2dict(args.annotations)
    if annotations['array_location'] == 'M1/PMd':
        seg = blocks[0].segments[1]
        area = 'MOTOR'
    else:
        seg = blocks[1].segments[1]
        area = 'VISION'

    all_trials = seg.filter(name='All Trials', objects=neo.Epoch)[0]
    if args.trial is None:
        trial = all_trials
    else:
        trial = all_trials[args.trial:args.trial+1]
        annotations.update(trial=args.trial)
    trial_seg = neo_utils.cut_segment_by_epoch(seg, trial)[0]

    asig = trial_seg.analogsignals[3]
    if not annotations['array_location'] == 'M1/PMd':
        asig = asig[:, asig.array_annotations['implantation_site'] == annotations['array_location']]

    asig = time_slice(asig, args.t_start, args.t_stop)

    # add metadata
    if args.annotations is not None:
        asig.annotations.update(parse_string2dict(args.annotations))

    asig.annotations.update(orientation_top=args.orientation_top)
    asig.annotations.update(orientation_right=args.orientation_right)

    kwargs = parse_string2dict(args.kwargs)

    channels = asig.array_annotations[kwargs['ELECTRODE_ANNOTATION_NAME']]

    coords = np.array([kwargs[f'NAME2COORDS_{area}'][str(channel)]
                       for channel in channels])

    asig.array_annotations.update(x_coords=coords[:,0])
    asig.array_annotations.update(y_coords=coords[:,1])

    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)

    # Save data
    block = neo.Block(name=args.data_name)
    seg = neo.Segment('Segment 1')
    seg.annotations = trial_seg.annotations
    block.segments.append(seg)
    block.segments[0].description = 'Loaded with neo.NixIO ' \
                         + f'(neo {neo.__version__}, nixio {nixio.__version__})'
    if asig.description is None:
        asig.description = ''
    block.segments[0].analogsignals.append(asig)

    # Save data
    if len(block.segments[0].analogsignals) > 1:
        raise Warning('Additional AnalogSignal found. The pipeline can yet \
                       only process single AnalogSignals.')

    with neo.NixIO(args.output) as io:
        io.write(block)
