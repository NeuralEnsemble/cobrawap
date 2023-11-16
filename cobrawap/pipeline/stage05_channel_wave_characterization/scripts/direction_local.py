"""
Calculate the wave directions per wave and channel, 
based on the spatial gradient of wave trigger times.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_str

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=str, required=True,
                    help="path to spatial derivative dataframe")
CLI.add_argument("--output", nargs='?', type=str, required=True,
                    help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                    help="path of output image file")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                    help="name of neo.Event to analyze (must contain waves)")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    df = pd.read_csv(args.data)

    direction_df = pd.DataFrame(df.channel_id, columns=['channel_id'])
    direction_df['direction_local_x'] = df.dt_x
    direction_df['direction_local_y'] = df.dt_y
    direction_df[f'{args.event_name}_id'] = df[f'{args.event_name}_id']

    direction_df.to_csv(args.output)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(np.angle(df.dt_x + 1j*df.dt_y), bins=36, range=[-np.pi, np.pi])

    if args.output_img is not None:
        save_plot(args.output_img)
