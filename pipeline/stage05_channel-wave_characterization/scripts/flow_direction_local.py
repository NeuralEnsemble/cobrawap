"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io import load_neo, save_plot
from utils.parse import none_or_str


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to neo object")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                     help="name of neo.Event to analyze (must contain waves)")
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    evt = block.filter(name=args.event_name, objects="Event")[0]
    evt = evt[evt.labels.astype('str') != '-1']

    optical_flow = block.filter(name='optical_flow', objects="AnalogSignal")[0]

    df_dict = {f'{args.event_name}_id': evt.labels,
               'channel_id': evt.array_annotations['channels'],
               'flow_direction_local_x': np.empty(len(evt), dtype=float),
               'flow_direction_local_y': np.empty(len(evt), dtype=float),
               }

    for i, trigger in enumerate(evt):
        t_idx = np.argmax(optical_flow.times >= trigger)
        channel_id = evt.array_annotations['channels'][i],
        direction = optical_flow[t_idx, channel_id]
        df_dict['flow_direction_local_x'][i] = np.real(direction)
        df_dict['flow_direction_local_y'][i] = np.imag(direction)

    df = pd.DataFrame(df_dict)
    df.to_csv(args.output)

    if args.output_img is not None:
        save_plot(args.output_img)
