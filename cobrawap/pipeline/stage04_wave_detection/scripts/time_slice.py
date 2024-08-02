"""
Cut data according to a start and stop time.
"""

import argparse
from pathlib import Path
import quantities as pq
from utils.io_utils import load_neo, write_neo
from utils.neo_utils import time_slice
from utils.parse import none_or_float

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=0,
                 help="new starting time in s")
CLI.add_argument("--t_stop", nargs='?', type=none_or_float, default=10,
                 help="new stopping time in s")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    for i, asig in enumerate(block.segments[0].analogsignals):
        block.segments[0].analogsignals[i] = time_slice(asig,
                                                        t_start=args.t_start,
                                                        t_stop=args.t_stop)

    for i, evt in enumerate(block.segments[0].events):
        block.segments[0].events[i] = time_slice(evt,
                                                 t_start=args.t_start,
                                                 t_stop=args.t_stop)


    write_neo(args.output, block)
