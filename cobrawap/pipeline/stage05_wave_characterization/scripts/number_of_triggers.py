"""
Calculate the number of triggers involved in each wave.
"""

import numpy as np
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
                 help="path of output image file")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name=args.event_name, objects="Event")[0]

    wave_ids = np.sort(np.unique(evts.labels).astype(int))

    number_of_triggers = np.empty(len(wave_ids), dtype=float)

    for i, wave_id in enumerate(wave_ids):
        idx = np.where(evts.labels == str(wave_id))[0]
        number_of_triggers[i] = len(evts.times[idx])

    # transform to DataFrame
    df = pd.DataFrame(number_of_triggers, columns=['number_of_triggers'])
    df[f'{args.event_name}_id'] = wave_ids
    df.to_csv(args.output)

    # ToDo
    if args.output_img is not None:
        save_plot(args.output_img)
