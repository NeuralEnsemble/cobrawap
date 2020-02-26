import neo
import numpy as np
import argparse

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--threshold", nargs='?', type=float)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        asig = io.read_block().segments[0].analogsignals[0]

    dim_t, channel_num = asig.shape

    np.save(args.output, np.ones(channel_num) * args.threshold)
