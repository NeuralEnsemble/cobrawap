import neo
import numpy as np
import quantities as pq
import argparse


def threshold(asig, threshold_array):
    dim_t, channel_num = asig.shape
    th_signal = asig.as_array()\
              - np.repeat(threshold_array[np.newaxis, :], dim_t, axis=0)
    state_array = th_signal > 0
    rolled_state_array = np.roll(state_array, 1, axis=0)

    all_times = np.array([])
    all_channels = np.array([])
    all_labels = np.array([])
    for label, func in zip(['UP',        'DOWN'],
                           [lambda x: x, lambda x: np.bitwise_not(x)]):
        trans = np.where(func(np.bitwise_not(rolled_state_array))\
                       * func(state_array))
        channels = trans[1]
        times = asig.times[trans[0]]

        if not len(times):
            raise ValueError("The choosen threshold lies not within the range "\
                           + "of the signal values!")

        all_channels = np.append(all_channels, channels)
        all_times = np.append(all_times, times)
        all_labels = np.append(all_labels, np.array([label for _ in times]))

    sort_idx = np.argsort(all_times)

    return neo.Event(times=all_times[sort_idx]*asig.times.units,
                     labels=all_labels[sort_idx],
                     name='Transitions',
                     array_annotations={'channels':all_channels[sort_idx]},
                     threshold=threshold_array)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--thresholds", nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    asig = block.segments[0].analogsignals[0]

    transition_event = threshold(asig, np.load(args.thresholds))

    block.segments[0].events.append(transition_event)

    with neo.NixIO(args.output) as io:
        io.write(block)
