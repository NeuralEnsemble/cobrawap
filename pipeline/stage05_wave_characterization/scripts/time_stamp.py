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
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image")
    CLI.add_argument("--time_point", nargs='?', type=str, default='start',
                     help="when to register the time for a wave [start, middle, end]")
    args = CLI.parse_args()

    block = load_neo(args.data)

    if args.time_point == 'start':
        time_stamp_func = np.min
    elif args.time_point == 'end':
        time_stamp_func = np.max
    elif args.time_point == 'middle':
        time_stamp_func = np.mean
    else:
        raise InputError('')

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]

    wave_ids = np.sort(np.unique(evts.labels).astype(int))
    if wave_ids[0] == -1:
        wave_ids = np.delete(wave_ids, 0)

    time_stamps = np.empty(len(wave_ids), dtype=np.float)

    t_unit = evts.times[0].dimensionality.string

    for i, wave_id in enumerate(wave_ids):
        idx = np.where(evts.labels == str(wave_id))[0]
        time_stamps[i] = time_stamp_func(evts.times[idx])


    fig, ax = plt.subplots(figsize=(15,2))
    for i, wave_id in enumerate(wave_ids):
        idx = np.where(evts.labels == str(wave_id))[0]
        t0, t1 = np.min(evts.times[idx]), np.max(evts.times[idx])
        ax.plot([t0,t1], [1,1], marker='|', color='b')
    ax.set_ylim((0,2))
    ax.set_xlabel(f'time [{t_unit}]')
    ax.set_title('wave occurences')
    save_plot(args.output_img)

    # transform to DataFrame
    df = pd.DataFrame(time_stamps,
                      columns=['time_stamp'],
                      index=wave_ids)
    df['time_stamp_unit'] = [t_unit]*len(wave_ids)
    df.index.name = 'wave_id'

    df.to_csv(args.output)
