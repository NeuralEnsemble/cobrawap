"""
Remove detected triggers (state transitions) when the corresponding Up and Down
states are shorter than a minimum duration.
"""

import numpy as np
import neo
import argparse
from pathlib import Path
import quantities as pq
from utils.io_utils import load_neo, write_neo
from utils.parse import str_to_bool

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path to output data in neo format")
CLI.add_argument("--min_up_duration", nargs='?', type=float, default=0.005,
                 help="minimum duration of UP states in seconds")
CLI.add_argument("--min_down_duration", nargs='?', type=float, default=0.005,
                 help="minimum duration of DOWN states in seconds")
CLI.add_argument("--remove_down_first", nargs='?', type=str_to_bool, default=True,
                 help="If True, remove short down states first")


def remove_short_states(evt, min_duration, start_label='UP', stop_label='DOWN'):
    # assumes event times to be sorted
    del_idx = np.array([], dtype=int)

    for channel in np.unique(evt.array_annotations['channels']):
        # select channel
        c_idx = np.where(channel == evt.array_annotations['channels'])[0]
        c_times = evt.times[c_idx]
        c_labels = evt.labels[c_idx]

        # sepearate start and stop times
        start_idx = np.where(start_label == c_labels)[0]
        stop_idx = np.where(stop_label == c_labels)[0]
        start_times = c_times[start_idx]
        stop_times = c_times[stop_idx]

        # clean borders
        leading_stops = np.argmax(stop_times > start_times[0])
        stop_idx = stop_idx[leading_stops:]
        stop_times = stop_times[leading_stops:]
        start_times = start_times[:len(stop_times)]

        # find short states
        short_state_idx = np.where((stop_times-start_times).rescale('s')
                                    < min_duration.rescale('s'))[0]

        # remove end points of short states
        del_idx = np.append(del_idx, c_idx[stop_idx[short_state_idx]])
        if not start_label == stop_label:
            # remove start points of short states
            del_idx = np.append(del_idx, c_idx[start_idx[short_state_idx]])

    cleaned_evt = neo.Event(times=np.delete(evt.times.rescale('s'), del_idx)*pq.s,
                            labels=np.delete(evt.labels, del_idx),
                            name=evt.name,
                            description=evt.description)
    cleaned_evt.annotations = evt.annotations
    for key in evt.array_annotations:
        cleaned_evt.array_annotations[key] = np.delete(evt.array_annotations[key],
                                                       del_idx)
    return cleaned_evt


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    evt_idx, evt = [(i,ev) for i, ev in enumerate(block.segments[0].events)
                    if ev.name == 'transitions'][0]


    if 'DOWN' in evt.labels:
        if args.remove_down_first:
            evt = remove_short_states(evt, args.min_down_duration*pq.s,
                                      start_label='DOWN', stop_label='UP')
            evt = remove_short_states(evt, args.min_up_duration*pq.s,
                                      start_label='UP', stop_label='DOWN')
        else:
            evt = remove_short_states(evt, args.min_up_duration*pq.s,
                                      start_label='UP', stop_label='DOWN')
            evt = remove_short_states(evt, args.min_down_duration*pq.s,
                                      start_label='DOWN', stop_label='UP')
    else:
        remove_short_states(evt, (args.min_down_duration+args.min_up_duration)*pq.s,
                            start_label='UP', stop_label='UP')

    evt.annotations.update(min_up_duration=args.min_up_duration*pq.s)
    evt.annotations.update(min_down_duration=args.min_down_duration*pq.s)

    block.segments[0].events[evt_idx] = evt

    write_neo(args.output, block)
