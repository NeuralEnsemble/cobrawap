import os
import sys
import argparse
import quantities as pq
import neo
import numpy as np
import scipy

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--output",  nargs='?', type=str)
    CLI.add_argument("--t_start",    nargs='?', type=float)
    CLI.add_argument("--t_stop",  nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        blk = io.read_block()


    # ToDo

    with neo.NixIO(args.output) as io:
        io.write(blk)
