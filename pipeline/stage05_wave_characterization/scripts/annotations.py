import neo
import os
import argparse
import numpy as np
import pandas as pd
import re
from utils.io import load_neo, save_plot
from utils.parse import none_or_str
from utils.neo import remove_annotations


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                     help="name of neo.Event to analyze (must contain waves)")
    CLI.add_argument("--ignore_keys", "--IGNORE_KEYS", nargs='+', type=str, default=[],
                     help="neo.Event annotations keys to not include in dataframe")
    args, unknown = CLI.parse_known_args()
    args.ignore_keys = [re.sub('[\[\],\s]', '', key) for key in args.ignore_keys]

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name=args.event_name, objects="Event")[0]

    remove_annotations(evts, del_keys=['nix_name', 'neo_name']+args.ignore_keys)

    ids = np.sort(np.unique(evts.labels).astype(int))
    df = pd.DataFrame(index=ids)
    df.index.name = f'{args.event_name}_id'

    for key, value in evts.annotations.items():
        df[key] = [value] * len(ids)

    df.to_csv(args.output)

    # ToDo
    save_plot(args.output_img)
