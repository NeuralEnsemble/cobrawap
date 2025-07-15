"""
Filter between a highpass and a lowpass frequency using a butterworth filter.
"""

import argparse
from pathlib import Path
import quantities as pq
import os
from elephant.signal_processing import butter
from utils.io_utils import load_neo, write_neo
from utils.parse import none_or_float

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--highpass_frequency", nargs='?', type=none_or_float,
                 default=None, help="lower bound of frequency band in Hz")
CLI.add_argument("--lowpass_frequency", nargs='?', type=none_or_float,
                 default=None, help="upper bound of frequency band in Hz")
CLI.add_argument("--order", nargs='?', type=int, default=2,
                 help="order of the filter function")
CLI.add_argument("--filter_function", nargs='?', type=str, default='filtfilt',
                 help="filter function used in the scipy backend")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = butter(block.segments[0].analogsignals[0],
                  highpass_frequency=args.highpass_frequency*pq.Hz,
                  lowpass_frequency=args.lowpass_frequency*pq.Hz,
                  order=args.order,
                  filter_function=args.filter_function)

    asig.array_annotations = block.segments[0].analogsignals[0].array_annotations
    asig.annotate(highpass_frequency=args.highpass_frequency*pq.Hz,
                  lowpass_frequency=args.lowpass_frequency*pq.Hz,
                  filter_order=args.order)

    asig.description += "Frequency filtered with [{}, {}]Hz order {} "\
                        .format(args.highpass_frequency,
                                args.lowpass_frequency,
                                args.order)\
                      + " using {} scipy algorithm.({}). "\
                        .format(args.filter_function,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
