import neo
import numpy as np
from itertools import groupby
import quantities as pq
import argparse


def threshold(asig, threshold_array):
    dim_t, channel_num = asig.shape
    th_signal = asig.as_array()\
              - np.repeat(threshold_array[np.newaxis, :], dim_t, axis=0)
    state_array = th_signal > 0
    rolled_state_array = np.roll(state_array, 1, axis=0)

    trans_events = {'UP': [], 'DOWN': []}
    for label, func in zip(['UP',        'DOWN'],
                           [lambda x: x, lambda x: np.bitwise_not(x)]):
        trans = np.where(func(np.bitwise_not(rolled_state_array))
                       * func(state_array))
        sort_idx = np.argsort(trans[1])
        trans_time_idx = trans[0][sort_idx]
        i = 0
        for channel, num_trans in [(k,len(list(g))) for k,g \
                                   in groupby(trans[1][sort_idx])]:
            evt = neo.Event(asig.times[np.sort(trans_time_idx[i:i+num_trans])],
                            labels=np.array([label for _ in range(num_trans)]),
                            name=channel,
                            threshold=threshold_array[channel])
            i += num_trans
            trans_events[label].append(evt)
    # merge events
    return trans_events['UP'], trans_events['DOWN']


def merge_events(up_events, down_events):
    events = []
    for ups, downs in zip(up_events, down_events):
        assert ups.name == downs.name
        times = np.append(ups.times, downs.times) * ups.times.units
        labels = np.append(ups.labels, downs.labels)
        sort_idx = np.argsort(times)
        evt = neo.Event(times[sort_idx],
                        labels=labels[sort_idx],
                        name=ups.name,
                        **ups.annotations)
        events.append(evt)
    return events


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--thresholds", nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    asig = block.segments[0].analogsignals[0]

    up_events, down_events = threshold(asig, np.load(args.thresholds))

    events = merge_events(up_events, down_events)

    prev_events = block.segments[0].events
    block.segments[0].events = events + prev_events

    with neo.NixIO(args.output) as io:
        io.write(block)
