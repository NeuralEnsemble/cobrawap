import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin
import argparse
from utils import load_neo, write_neo, remove_annotations


def detect_minima(asig, order):
    signal = asig.as_array()
    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)

    sort_idx = np.argsort(t_idx)

    evt = neo.Event(times=asig.times[t_idx[sort_idx]],
                     labels=['UP'] * len(t_idx),
                     name='Transitions',
                     minima_order=order,
                     array_annotations={'channels':channel_idx[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)
    return evt

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--order", nargs='?', type=int, default=3,
                     help="number of neighbouring points to compare")
    args = CLI.parse_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig, args.order)

    block.segments[0].events.append(transition_event)

    write_neo(args.output, block)
