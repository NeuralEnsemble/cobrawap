import argparse
import quantities as pq
import numpy as np
from elephant.signal_processing import zscore
from utils.io import load_neo, write_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--target_rate",  nargs='?', type=float, required=True,
                     help="rate to subsample to in Hz")

    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    subsampling_order = asig.sampling_rate/(args.target_rate*pq.Hz)
    subsampling_order = int(np.round(subsampling_order.rescale('dimensionless')))

    sub_asig = asig.duplicate_with_new_data(asig.as_array()[::subsampling_order])
    sub_asig.sampling_rate = asig.sampling_rate/subsampling_order

    sub_asig.array_annotations = asig.array_annotations
    block.segments[0].analogsignals[0] = sub_asig

    write_neo(args.output, block)
