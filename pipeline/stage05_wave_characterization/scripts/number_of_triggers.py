import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import scipy
import pandas as pd
from utils import load_neo, none_or_str, save_plot


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    args = CLI.parse_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]

    wave_ids = np.sort(np.unique(evts.labels).astype(int))

    number_of_triggers = np.empty(len(wave_ids), dtype=np.float)

    for i, wave_id in enumerate(wave_ids):
        idx = np.where(evts.labels == str(wave_id))[0]
        number_of_triggers[i] = len(evts.times[idx])

    # transform to DataFrame
    df = pd.DataFrame(number_of_triggers,
                      columns=['number_of_triggers'],
                      index=wave_ids)
    df.index.name = 'wave_id'

    df.to_csv(args.output)
