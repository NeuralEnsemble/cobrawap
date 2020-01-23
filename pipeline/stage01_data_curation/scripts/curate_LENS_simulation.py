import numpy as np
import argparse
import neo
import quantities as pq
import json
import os
import sys
import scipy
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import parse_string2dict, ImageSequence2AnalogSignal

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
    CLI.add_argument("--sampling_rate", nargs='?', type=float)
    CLI.add_argument("--spatial_scale", nargs='?', type=float)
    CLI.add_argument("--t_start", nargs='?', type=none_or_float)
    CLI.add_argument("--t_stop", nargs='?', type=none_or_float)
    CLI.add_argument("--data_name", nargs='?', type=str)
    CLI.add_argument("--annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str)
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str)
    args = CLI.parse_args()

    mat = scipy.io.loadmat(args.data)

    asig = neo.AnalogSignal(mat['NuE'].T,
                            units='dimensionless',
                            t_start=0*pq.s,
                            sampling_rate=args.sampling_rate*pq.Hz,
                            spatial_scale=args.spatial_scale*pq.mm,
                            )

    asig.array_annotate(x_coords=np.squeeze(mat['x_pos_sel']),
                        y_coords=np.squeeze(mat['y_pos_sel']))

    if args.t_start is not None or args.t_stop is not None:
        if args.t_start is None:
            args.t_start == asig.t_start.rescale('s').magnitude
        if args.t_stop is None:
                args.t_stop == asig.t_stop.rescale('s').magnitude
        asig = asig.time_slice(t_start=args.t_start*pq.s,
                               t_stop=args.t_stop*pq.s)

    block = neo.Block()
    seg = neo.Segment()
    block.segments.append(seg)
    block.segments[0].analogsignals.append(asig)
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    if block.segments[0].analogsignals[0].description is None:
        block.segments[0].analogsignals[0].description = ''
    block.segments[0].analogsignals[0].description += 'simulated Ca+ imaging signal. '

    # Save data
    with neo.NixIO(args.output) as io:
        io.write(block)
