"""
Plot the critical points in the optical flow vector field.
"""

import argparse
from pathlib import Path
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.neo_utils import analogsignal_to_imagesequence

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--skip_step", nargs='?', type=int, default=3,
                 help="skipping every x vector for the plot")
CLI.add_argument("--frame_id", nargs='?', type=int, default=0,
                 help="number of the frame to plot")

def plot_frame(frame, ax=None, skip_step=3):
    dim_y, dim_x = frame.shape

    if ax is None:
        fig, ax = plt.subplots()

    ax.quiver(np.arange(dim_x)[::skip_step],
              np.arange(dim_y)[::skip_step],
              np.real(frame[::skip_step,::skip_step]),
              np.imag(frame[::skip_step,::skip_step]))

    Y, X = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
    ZR = np.real(frame)
    ZI = np.imag(frame)
    contourR = ax.contour(X, Y, ZR, levels=[0], colors='b', label='x = 0')
    contourI = ax.contour(X, Y, ZI, levels=[0], colors='g', label='y = 0')
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()
    block = load_neo(args.data)

    asigs = block.filter(name='optical_flow', objects="AnalogSignal")

    if asigs:
        asig = asigs[0]
    else:
        raise ValueError("Input does not contain a signal with name " \
                       + "'optical_flow'!")

    imgseq = analogsignal_to_imagesequence(asig)

    crit_point_evt = [evt for evt in block.segments[0].events
                      if evt.name == "critical_points"]
    if crit_point_evt:
        crit_point_evt = crit_point_evt[0]
    else:
        raise ValueError("Input does not contain an event with name " \
                       + "'critical_points'!")

    fig, ax = plt.subplots()

    ax = plot_frame(imgseq.as_array()[args.frame_id],
                    skip_step=args.skip_step,
                    ax=ax)

    start_id = np.argmax(crit_point_evt.times >= asig.times[args.frame_id])
    stop_id = np.argmax(crit_point_evt.times >= asig.times[args.frame_id+1])

    ax.scatter(crit_point_evt.array_annotations['x'][start_id:stop_id],
               crit_point_evt.array_annotations['y'][start_id:stop_id],
               color='r')

    ax.set_title(f"{asig.times[args.frame_id]:.2f}")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    save_plot(args.output)
