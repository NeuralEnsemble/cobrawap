"""
Extract the annotations of Neo objects and structure them in a DataFrame
to complement a wave characterization.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import quantities as pq
import re
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path, none_or_str
from utils.neo_utils import remove_annotations

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")
CLI.add_argument("--ignore_keys", "--IGNORE_KEYS", nargs='*', type=str, default=[],
                 help="neo object annotations keys to not include in dataframe")
CLI.add_argument("--include_keys", "--INCLUDE_KEYS", nargs='*', type=str, default=[],
                 help="neo object annotations keys to include in dataframe")
CLI.add_argument("--profile", "--PROFILE", nargs='?', type=none_or_str, default=None,
                 help="profile name")

def add_annotations_to_df(df, annotations, include_keys=[]):
    use_all_keys = not bool(len(include_keys))

    for key, value in annotations.items():
        key_is_relevant = use_all_keys or key in include_keys

        if key_is_relevant and key not in df.columns:
            if type(value) == pq.Quantity:
                df[f'{key}_unit'] = value.dimensionality.string
                value = value.magnitude
            df[key] = value

    return df

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()
    args.ignore_keys = [re.sub(r"[\[\],\s]", "", key) for key in args.ignore_keys]
    args.include_keys = [re.sub(r"[\[\],\s]", "", key) for key in args.include_keys]
    if len(args.include_keys):
        args.ignore_keys = []

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels.astype(str) != '-1']

    df = pd.DataFrame(evts.labels, columns=[f'{args.event_name}_id'])
    df['channel_id'] = evts.array_annotations['channels']
    args.ignore_keys += ['channels']

    remove_annotations(evts, del_keys=['nix_name', 'neo_name']+args.ignore_keys)
    remove_annotations(asig, del_keys=['nix_name', 'neo_name']+args.ignore_keys)

    for annotations in [evts.annotations, evts.array_annotations,
                        asig.annotations]:
        df = add_annotations_to_df(df, annotations, args.include_keys)

    df['profile'] = [args.profile] * len(df.index)
    df['sampling_rate'] = asig.sampling_rate.magnitude
    df['sampling_rate_unit'] = asig.sampling_rate.dimensionality.string
    df['recording_length'] = (asig.t_stop - asig.t_start).magnitude
    df['recording_length_unit'] = asig.t_start.dimensionality.string
    df['dim_x'] = int(max(asig.array_annotations['x_coords']))+1
    df['dim_y'] = int(max(asig.array_annotations['y_coords']))+1

    df.to_csv(args.output)

    # ToDo
    if args.output_img is not None:
        save_plot(args.output_img)
