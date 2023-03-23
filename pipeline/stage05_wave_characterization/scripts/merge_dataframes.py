"""
Merge pandas DataFrames based on the values of selected columns.
"""

import argparse
import pandas as pd
from copy import deepcopy

CLI = argparse.ArgumentParser()
CLI.add_argument("--output",    nargs='?', type=str)
CLI.add_argument("--data",      nargs='+', type=str)
CLI.add_argument("--output_img",nargs='?', type=str)
# CLI.add_argument("--merge_key", nargs='?', type=str)

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    for i, datafile in enumerate(args.data):
        df = pd.read_csv(datafile)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],
                axis=1, inplace=True)
        if i:
            full_df = full_df.merge(df, how='outer', on=None)
        else:
            full_df = deepcopy(df)
        del df

    full_df.to_html(args.output_img)

    full_df.to_csv(args.output)
