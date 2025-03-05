"""
Check whether the input data representation adheres to the stage's requirements.

Additionally prints a short summary of the data attributes.
"""

import numpy as np
import argparse
from pathlib import Path
from utils.io_utils import load_neo

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    if len(block.segments) > 1:
        print("More than one Segment found; all except the first one " \
            + "will be ignored.")
    if len(block.segments[0].analogsignals) > 1:
        print("More than one AnalogSignal found; all except the first one " \
            + "will be ignored.")

    asig = block.segments[0].analogsignals[0]

    print('Recording Time:\t\t', asig.t_stop - asig.t_start)
    print('Sampling Rate:\t\t', asig.sampling_rate)
    print('Spatial Scale:\t\t', asig.annotations['spatial_scale'])

    num_channels = np.count_nonzero(~np.isnan(np.sum(asig, axis=0)))
    print('Number of Channels:\t', num_channels)

    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1
    print('Grid Dimensions:\t', f'{dim_x} x {dim_y}')
    print('Empty Grid Sites:\t', dim_x*dim_y - num_channels)
