"""

"""
import numpy as np
import argparse
import quantities as pq
from utils.io import load_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    args = CLI.parse_args()

    block = load_neo(args.data)

    if len(block.segments) > 1:
        print("More than one Segment found; all except the first one " \
            + "will be ignored.")
    if len(block.segments[0].analogsignals) > 1:
        print("More than one AnalogSignal found; all except the first one " \
            + "will be ignored.")

    asig = block.segments[0].analogsignals[0]

    # print('Recording Time:\t\t', asig.t_stop - asig.t_start)
    # print('Sampling Rate:\t\t', asig.sampling_rate)
    # print('Spatial Scale:\t\t', asig.annotations['spatial_scale'])

    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if not len(evts):
        raise ValueError("No 'Transitions' events found!")
    evt = evts[0]

    if not 'UP' in evt.labels:
        raise KeyError("No transitions labeled 'UP' found!")

    up_channels = np.unique(evt.array_annotations['channels'])
    num_channels = np.count_nonzero(~np.isnan(np.sum(asig, axis=0)))
    print(f'{len(up_channels)} of {num_channels} channels show UP transitions.')

    evt.array_annotations['x_coords']
    evt.array_annotations['y_coords']
