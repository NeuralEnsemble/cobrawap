"""
Set the threshold between Up and Down states to a fixed value.
"""

import numpy as np
import argparse
from pathlib import Path
from utils.io_utils import load_neo

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output thresholds (numpy array)")
CLI.add_argument("--threshold", nargs='?', type=float, required=True,
                 help="")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    asig = load_neo(args.data, 'analogsignal')

    dim_t, channel_num = asig.shape

    np.save(args.output, np.ones(channel_num) * args.threshold)
