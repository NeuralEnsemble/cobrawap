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
    if len(block.segments[0].analogsignals) > 1:
        print("More than one AnalogSignal found; all except the first one " \
            + "will be ignored.")

    asig = block.segments[0].analogsignals[0]

    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']

    print('Recording Time:\t\t', asig.t_stop - asig.t_start)
    print('Sampling Rate:\t\t', asig.sampling_rate)
    num_channels = np.count_nonzero(~np.isnan(np.sum(asig, axis=0)))
    print('Number of Channels:\t', num_channels)

    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1
    print('Grid Dimensions:\t', f'{dim_x} x {dim_y}')
    print('Empty Grid Sites:\t', dim_x*dim_y - num_channels)
