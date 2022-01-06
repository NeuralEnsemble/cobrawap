import numpy as np
from elephant.signal_processing import hilbert
import argparse
import os
from utils.io import load_neo, write_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    phase = np.angle(hilbert(asig).as_array())

    asig = asig.duplicate_with_new_data(phase)
    asig.array_annotations = block.segments[0].analogsignals[0].array_annotations

    asig.name += ""
    asig.description += "Phase signal ({}). "\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
