"""

"""
import numpy as np
import argparse
import quantities as pq
import warnings
from utils.io import load_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    args = CLI.parse_args()

    block = load_neo(args.data)

    if len(block.segments) > 1:
        print("More than one Segment found; all except the first one " \
            + "will be ignored.")

    evts = block.filter(name='Wavefronts', objects="Event")
    if not len(evts):
        raise ValueError("No 'Wavefronts' events found!")

    evt = evts[0]
    evt = evt[evt.labels != '-1']
    num_waves = len(np.unique(evt.labels))

    if num_waves:
        print(f'{num_waves} wavefronts found.')
    else:
        raise ValueError("There are no waves detected!")

    evt.array_annotations['x_coords']
    evt.array_annotations['y_coords']
    evt.annotations['spatial_scale']

    optical_flow = block.filter(name='Optical Flow', objects="AnalogSignal")
    if not len(evts):
        warnings.warn('No Optical-Flow signal available!')
