"""
Calculate the period between two consecutive waves for each wave.
"""

from pathlib import Path
import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import scipy
import pandas as pd
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='wavefronts', objects="Event")[0]
    num_nonnan_channels = np.sum(np.isfinite(asig[0]).astype(int))

    wave_ids = np.sort(np.unique(evts.labels).astype(int))
    if wave_ids[0] == -1:
        wave_ids = np.delete(wave_ids, 0)

    wave_times = np.empty((2, asig.shape[1]), dtype=float)

    IWIs = np.empty((len(wave_ids),2))
    IWIs.fill(np.nan)

    t_unit = evts.times[0].dimensionality.string

    for wave_id in wave_ids[:-1]:
        wave_times.fill(np.nan)
        for i, wi in enumerate([wave_id, wave_id+1]):
            idx = np.where(evts.labels == str(wi))[0]
            trigger_channels = evts.array_annotations['channels'][idx]
            trigger_times = evts.times[idx]
            for channel, t in zip(trigger_channels, trigger_times):
                wave_times[i, int(channel)] = t
        inter_wave_intervals = wave_times[1] - wave_times[0]
        IWIs[wave_id, 0] = np.nanmean(inter_wave_intervals)
        IWIs[wave_id, 1] = np.nanstd(inter_wave_intervals)

    # transform to DataFrame
    df = pd.DataFrame(IWIs, columns=['inter_wave_interval', 'inter_wave_interval_std'])
    df['inter_wave_interval_unit'] = t_unit
    df[f'{args.event_name}_id'] = wave_ids
    df.to_csv(args.output)

    # ToDo
    fig, ax = plt.subplots()

    if np.isfinite(IWIs[:,0]).any():
        ax.hist(IWIs[:,0])

    ax.set_xlabel('inter-wave interval [s]')
    if args.output_img is not None:
        save_plot(args.output_img)
