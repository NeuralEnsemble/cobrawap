"""

"""
import numpy as np
import argparse
import quantities as pq
from utils import load_neo, none_or_int


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

    evts = [ev for ev in block.segments[0].events if ev.name== 'Wavefronts']
    if not len(evts):
        raise ValueError("No 'Wavefronts' events found!")
    evt = evts[0]

    ids = np.unique(evt.labels)
    print(f'{len(ids)} wavefronts found.')

    evt.array_annotations['x_coords']
    evt.array_annotations['y_coords']
    evt.annotations['spatial_scale']
