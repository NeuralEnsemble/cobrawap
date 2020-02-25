"""
ToDo
"""
import numpy as np
import neo
import argparse
import os
from utils import load_neo, write_neo

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
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--order", nargs='?', type=int, default=2,
                     help="detrending order")
    args = CLI.parse_args()

    block = load_neo(args.data)

    asig = detrending(block.segments[0].analogsignals[0], args.order)

    asig.name += ""
    asig.description += "Detrended by order {} ({}). "\
                        .format(args.order, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
