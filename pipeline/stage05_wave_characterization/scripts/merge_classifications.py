import argparse
import numpy as np
import pandas as pd
from utils import load_neo, write_neo, AnalogSignal2ImageSequence


def merging_labels(dataframe_files):
    for i, wave_class_file in enumerate(dataframe_files):
        labels = pd.read_csv(wave_class_file, index_col=0)
        if i:
            wave_labels = pd.merge(wave_labels, labels, on='wave_id', how='outer')
        else:
            wave_labels = labels
    return wave_labels


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output data in neo format")
    CLI.add_argument("--labels", nargs='?', type=lambda v: v.split(','),
                     required=True, help="path(s) classification csv files")

    args = CLI.parse_args()

    wave_labels = merging_labels(args.labels)

    wave_labels.to_csv(args.output)
