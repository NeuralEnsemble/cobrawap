import argparse
import quantities as pq
from utils.io import load_neo, write_neo
from utils.neo import time_slice


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--t_start", nargs='?', type=float, default=0,
                     help="new starting time in s")
    CLI.add_argument("--t_stop", nargs='?', type=float, default=10,
                     help="new stopping time in s")
    args = CLI.parse_args()

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
