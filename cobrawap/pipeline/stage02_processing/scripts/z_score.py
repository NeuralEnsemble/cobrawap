"""
Z-score the signal of each channel.
"""

import argparse
from pathlib import Path
from elephant.signal_processing import zscore
from utils.io_utils import load_neo, write_neo

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    zscore(block.segments[0].analogsignals[0], inplace=True)

    write_neo(args.output, block)
