import numpy as np
import argparse
import neo
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape


def normalize(asig, normalize_by):
    if normalize_by == 'median':
        norm_function = np.median
    elif normalize_by == 'max':
        norm_function = np.max
    elif normalize_by == 'mean':
        norm_function = np.mean
    else:
        raise InputError("The method to normalize by is not recognized. "\
                       + "Please choose either 'mean', 'median', or 'max'.")

    dim_t, num_channels = asig.shape
    norm_asig = asig.as_array()
    for i in range(num_channels):
        norm_value = norm_function(norm_asig[:,i])
        if norm_value:
            norm_asig[:,i] /= norm_value
        else:
            print("Normalization factor is {} for channel {} "\
                  .format(nom_value, i) + "and was skipped.")
    for num in range(dim_t):
        asig[num] = norm_asig[num]
    del norm_asig
    return asig


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--normalize_by", nargs='?', type=str)
    args = CLI.parse_args()

    # load images
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)

    asig = normalize(block.segments[0].analogsignals[0], args.normalize_by)

    # save processed data
    asig.name += ""
    asig.description += "Normalized by {} ({})."\
                        .format(args.normalize_by, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
