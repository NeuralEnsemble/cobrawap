import numpy as np
from elephant.signal_processing import hilbert
import neo
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",        nargs='?', type=str)
    CLI.add_argument("--data",          nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)

    asig = block.segments[0].analogsignals[0]
    phase = np.angle(hilbert(asig).as_array())

    array_annotations = asig.array_annotations
    asig = asig.duplicate_with_new_data(phase)
    asig.array_annotations.update(**array_annotations)

    # save processed data
    asig.name += ""
    asig.description += "Phase signal ({}). "\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
