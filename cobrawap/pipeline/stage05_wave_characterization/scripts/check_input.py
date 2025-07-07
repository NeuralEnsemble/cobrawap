"""
Check whether the input data representation adheres to the stage's requirements.

Additionally prints a short summary of the data attributes.
"""

import numpy as np
import argparse
from pathlib import Path
import warnings
import re
from utils.io_utils import load_neo
from utils.parse import none_or_str

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=none_or_str, default=None,
                 help="name of neo.Event to analyze (must contain waves)")
CLI.add_argument("--measures", "--MEASURES", nargs='+', type=none_or_str, default=None,
                 help="list of measure names to apply")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    if args.measures is not None and args.event_name == 'wavemodes':
        mode_invalid = ['label_planar', 'inter_wave_interval',
                        'number_of_triggers', 'time_stamp']
        args.measures = [re.sub(r"[\[\],\s]", "", measure) for measure in args.measures]
        invalid_measures = [measure for measure in args.measures \
                                                if measure in mode_invalid]
        if len(invalid_measures):
            warnings.warn('The following selected measures are can not be '
                          'calculated for wavemodes and will be skipped: '
                         f'{", ".join(invalid_measures)}.')

    block = load_neo(args.data)

    if len(block.segments) > 1:
        print("More than one Segment found; all except the first one " \
            + "will be ignored.")

    evts = block.filter(name='wavefronts', objects="Event")
    if not len(evts):
        raise ValueError("No 'wavefronts' events found!")

    evt = evts[0]
    evt = evt[evt.labels != '-1']
    num_waves = len(np.unique(evt.labels))

    if num_waves:
        print(f'{num_waves} wavefronts found.')
    else:
        raise ValueError("There are no waves detected!")

    evt.array_annotations['x_coords']
    evt.array_annotations['y_coords']
    evt.annotations['spatial_scale']

    evts = block.filter(name='optical_flow', objects="AnalogSignal")
    if not len(evts):
        warnings.warn('No Optical-Flow signal available!')

    evts = block.filter(name='wavemodes', objects="Event")
    if len(evts):
        print(f'{len(np.unique(evts[0].labels))} wavemodes found')
    else:
        warnings.warn("No 'wavemodes' events found!")
