import numpy as np
import neo
import argparse
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape, remove_annotations



def detrending(asig, order):
    # ToDo: Improve algorithm and include into elephant
    if not type(order) == int:
        raise TypeError("Detrending order needs to be int.")
    if order < 0 or order > 4:
        raise InputError("Detrending order must be between 0 and 4!")

    dim_t, num_channels = asig.shape
    X = asig.as_array()
    window_size = len(asig)

    if order > 0:
        X = X - np.mean(X, axis=0)
    if order > 1:
        factor = [1, 1/2., 1/6.]
        for i in np.arange(order-1)+1:
            detrend = np.linspace(-window_size/2., window_size/2., window_size)**i \
                      * np.mean(np.diff(X, n=i, axis=0)) * factor[i-1]
            X = X - detrend

    for num in range(dim_t):
        asig[num] = X[num]

    return asig



if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--order", nargs='?', type=int)
    args = CLI.parse_args()

    # load images
    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)
    remove_annotations([block] + block.segments
                       + block.segments[0].analogsignals)

    asig = detrending(block.segments[0].analogsignals[0], args.order)

    # save processed data
    asig.name += ""
    asig.description += "Detrended by order {} ({})."\
                          .format(args.order, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
