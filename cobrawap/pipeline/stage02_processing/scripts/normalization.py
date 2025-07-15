"""
Divide the signal in each channel by their max/mean/median value.
"""

import numpy as np
import argparse
from pathlib import Path
import neo
import quantities as pq
import os
import sys
from utils.io_utils import write_neo, load_neo

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--normalize_by", nargs='?', type=str, default='mean',
                 help="division factor: 'max', 'mean', or 'median'")

def normalize(asig, normalize_by):
    if normalize_by == 'median':
        norm_function = np.median
    elif normalize_by == 'max':
        norm_function = np.max
    elif normalize_by == 'mean':
        norm_function = np.mean
    else:
        raise ValueError("The method to normalize by is not recognized. "\
                       + "Please choose either 'mean', 'median', or 'max'.")

    dim_t, num_channels = asig.shape
    norm_asig = asig.as_array()
    for i in range(num_channels):
        norm_value = norm_function(norm_asig[:,i])
        if norm_value:
            norm_asig[:,i] /= norm_value
        else:
            print("Normalization factor is {} for channel {} "\
                  .format(norm_value, i) + "and was skipped.")

    new_asig = asig.duplicate_with_new_data(norm_asig, units='dimensionless')
    new_asig.array_annotations = asig.array_annotations
    return new_asig


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = normalize(block.segments[0].analogsignals[0], args.normalize_by)

    asig.description += "Normalized by {} ({})."\
                        .format(args.normalize_by, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
