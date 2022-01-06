import neo
import numpy as np
import argparse
from utils.io import load_neo

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output thresholds (numpy array)")
    CLI.add_argument("--threshold", nargs='?', type=float, required=True)
    args = CLI.parse_args()

    asig = load_neo(args.data, 'analogsignal')

    dim_t, channel_num = asig.shape

    np.save(args.output, np.ones(channel_num) * args.threshold)
