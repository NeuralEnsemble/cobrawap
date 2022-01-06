import argparse
from elephant.signal_processing import zscore
from utils.io import load_neo, write_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")

    args = CLI.parse_args()

    block = load_neo(args.data)

    zscore(block.segments[0].analogsignals[0], inplace=True)

    write_neo(args.output, block)
