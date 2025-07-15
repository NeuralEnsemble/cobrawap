"""
Calculate the wave propagation velocity for each wave and channel.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to spatial derivative dataframe")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    df = pd.read_csv(args.data)

    velocity = df.spatial_scale * np.sqrt(1/(df.dt_x**2 + df.dt_y**2))
    velocity[~np.isfinite(velocity)] = np.nan

    velocity_df = pd.DataFrame(velocity, columns=['velocity_local'])
    velocity_df['channel_id'] = df.channel_id
    velocity_df['velocity_local_unit'] = f'{df.spatial_scale_unit[0]}/{df.dt_unit[0]}'
    velocity_df[f'{args.event_name}_id'] = df[f'{args.event_name}_id']

    velocity_df.to_csv(args.output)

    if args.output_img is not None:
        save_plot(args.output_img)
