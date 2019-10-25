import numpy as np
import neo
import argparse
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape, remove_annotations



def detrending(signal, order):
    if isinstance(signal, neo.AnalogSignal):
        X = signal.as_array()
    elif isinstance(signal, pq.quantity.Quantity):
        X = copy(signal.magnitude)
    elif isinstance(signal, np.ndarray):
        X = copy(signal)
    else:
        raise TypeError('Input signal must be either an AnalogSignal,'
                      + 'a quantity array, or a numpy array.')
    if not type(order) == int:
        raise TypeError("Detrending order needs to be int.")
    if order < 0 or order > 4:
        raise InputError("Detrending order must be between 0 and 4!")

    dim_t, num_channels = signal.shape
    window_size = len(signal)

    if order > 0:
        X = X - np.mean(X, axis=0)
    if order > 1:
        factor = [1, 1/2., 1/6.]
        for i in np.arange(2, order):
            detrend = np.zeros_like(X)
            for channel, x in enumerate(X.T):
                detrend[:,channel] =\
                    np.linspace(-window_size/2., window_size/2., window_size)**i \
                    * np.mean(np.diff(x, n=i)) * factor[i-2]
            X = X - detrend

    if isinstance(signal, neo.AnalogSignal):
        signal_out = signal.duplicate_with_new_data(X)
        signal_out.array_annotate(**signal.array_annotations)
        return signal_out
    elif isinstance(signal, pq.quantity.Quantity):
        return X * signal.units
    elif isinstance(signal, np.ndarray):
        return X



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
    asig.description += "Detrended by order {} ({}). "\
                        .format(args.order, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
