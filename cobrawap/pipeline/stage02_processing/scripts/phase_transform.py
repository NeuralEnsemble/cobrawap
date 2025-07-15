"""
Replace the data signal value with their corresponding Hilbert phase.
"""

import numpy as np
from elephant.signal_processing import hilbert
import argparse
from pathlib import Path
import os
from utils.io_utils import load_neo, write_neo

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    phase = np.angle(hilbert(asig).as_array())

    asig = asig.duplicate_with_new_data(phase)
    asig.array_annotations = block.segments[0].analogsignals[0].array_annotations

    asig.description += "Phase signal ({}). "\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
