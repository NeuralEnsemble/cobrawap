import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin
import argparse
import matplotlib.pyplot as plt


def detect_minima(asig, order):
    signal = asig.as_array()

    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)

    sort_idx = np.argsort(t_idx)

    return neo.Event(times=asig.times[t_idx[sort_idx]],
                     labels=['UP'] * len(t_idx),
                     name='Transitions',
                     array_annotations={'channels':channel_idx[sort_idx]})


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--order",      nargs='?', type=int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig, args.order)

    block.segments[0].events.append(transition_event)

    with neo.NixIO(args.output) as io:
        io.write(block)
