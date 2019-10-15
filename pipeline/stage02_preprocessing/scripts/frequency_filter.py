import numpy as np
from elephant.signal_processing import butter
import matplotlib.pyplot as plt
import argparse
import neo
import os
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape, remove_annotations


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--highpass_freq", nargs='?', type=none_or_float)
    CLI.add_argument("--lowpass_freq", nargs='?', type=none_or_float)
    CLI.add_argument("--order", nargs='?', type=int)
    CLI.add_argument("--filter_function", nargs='?', type=str)
    args = CLI.parse_args()

    # load images
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    remove_annotations([block] + block.segments
                       + block.segments[0].analogsignals)

    asig = butter(block.segments[0].analogsignals[0],
                  highpass_freq=args.highpass_freq,
                  lowpass_freq=args.lowpass_freq,
                  order=args.order,
                  filter_function=args.filter_function,
                  axis=0)

    # save processed data
    asig.name += ""
    asig.description += "Frequency filtered with [{}, {}]Hz order {} "\
                      + " using {} scipy algorithm.({})."\
                        .format(args.highpass_freq,
                                args.lowpass_freq,
                                args.order,
                                args.filter_function,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
