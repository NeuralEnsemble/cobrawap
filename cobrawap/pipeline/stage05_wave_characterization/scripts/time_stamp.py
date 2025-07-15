"""
Calculate the timing of each wave.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pandas as pd
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image")
CLI.add_argument("--time_point", "--TIME_STAMP_POINT", nargs='?', type=str, default='start',
                 help="when to register the time for a wave [start, middle, end]")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

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
    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels.astype('str') != '-1']

    wave_ids = np.sort(np.unique(evts.labels).astype(int))

    time_stamps = np.empty(len(wave_ids), dtype=float)

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
    if args.output_img is not None:
        save_plot(args.output_img)

    # transform to DataFrame
    df = pd.DataFrame(time_stamps, columns=['time_stamp'])
    df['time_stamp_unit'] = t_unit
    df[f'{args.event_name}_id'] = wave_ids

    df.to_csv(args.output)
