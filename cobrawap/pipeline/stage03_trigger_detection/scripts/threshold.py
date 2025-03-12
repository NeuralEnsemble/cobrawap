"""
Detect trigger times (i.e., state transition / local wavefronts onsets)
by applying a threshold to each channel signal.
"""

import neo
import numpy as np
import argparse
from pathlib import Path
from utils.io_utils import load_neo, write_neo
from utils.neo_utils import remove_annotations

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--thresholds", nargs='?', type=str, required=True,
                 help="path of thresholds (numpy array)")

def threshold(asig, threshold_array):
    dim_t, channel_num = asig.shape
    th_signal = asig.as_array()\
              - np.repeat(threshold_array[np.newaxis, :], dim_t, axis=0)
    state_array = th_signal > 0
    rolled_state_array = np.roll(state_array, 1, axis=0)

    all_times = np.array([])
    all_channels = np.array([], dtype=int)
    all_labels = np.array([])
    for label, func in zip(['UP',        'DOWN'],
                           [lambda x: x, lambda x: np.bitwise_not(x)]):
        trans = np.where(func(np.bitwise_not(rolled_state_array))\
                       * func(state_array))
        channels = trans[1]
        times = asig.times[trans[0]]

        if not len(times):
            raise ValueError("The chosen threshold lies not within the range "\
                           + "of the signal values!")

        all_channels = np.append(all_channels, channels)
        all_times = np.append(all_times, times)
        all_labels = np.append(all_labels, np.array([label for _ in times]))

    sort_idx = np.argsort(all_times)

    evt = neo.Event(times=all_times[sort_idx]*asig.times.units,
                    labels=all_labels[sort_idx],
                    name='transitions',
                    trigger_detection='threshold',
                    array_annotations={'channels':all_channels[sort_idx]},
                    threshold=threshold_array,
                    description='Transitions between down and up states with '\
                               +'labels "UP" and "DOWN". '\
                               +'Annotated with the channel id ("channels").')

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][all_channels[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)
    return evt


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]

    transition_event = threshold(asig, np.load(args.thresholds))

    block.segments[0].events.append(transition_event)

    write_neo(args.output, block)
