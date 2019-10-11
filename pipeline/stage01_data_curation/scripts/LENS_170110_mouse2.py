import numpy as np
import argparse
import neo
import os
import sys
sys.path.append('../../')
from utils import str2dict, remove_annotations

def ImageSequence2AnalogSignal(imagesequence):

    return analogsignal


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--sampling_rate", nargs='?', type=float)
    CLI.add_argument("--pixel_size", nargs='?', type=float)
    CLI.add_argument("--data_name", nargs='?', type=str)
    CLI.add_argument("--annotations", nargs='+', type=str)
    args = CLI.parse_args()

    io = neo.io.tiffio.TiffIO(directory_path=args.data)

    block = io.read_block(sampling_rate=args.sampling_rate*pq.Hz,
                          spatial_scale=args.pixel_size*pq.mm,
                          units='dimensionless')

    remove_annotations([block] + block.segments
                       + block.segments[0].analogsignals)


    # ToDo: ImageSequence to AnalogSignal

    # ToDo: annotate AnalogSignal

    block.name = args.data_name
    # block.segments[0].name =
    # block.segments[0].description =

    # Save as NIX file

    with neo.NixIO(args.output) as io:
        io.write(block)
