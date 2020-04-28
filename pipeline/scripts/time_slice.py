import argparse
from utils import load_neo, write_neo, time_slice


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

    for i, asig in enumerate(blk.segments[0].analogsignals):
        blk.segments[0].analogsignals[i] = time_slice(asig, t_start=t_start,
                                                      t_stop=t_stop)

    for i, evt in enumerate(blk.segments[0].events):
        blk.segments[0].events[i] = time_slice(evt, t_start=t_start,
                                               t_stop=t_stop)


    write_neo(args.output, block)
