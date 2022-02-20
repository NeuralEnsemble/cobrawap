import argparse
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from utils.io import load_neo, write_neo, save_plot
from utils.neo import analogsignals_to_imagesequences


def plot_frame(frame, ax=None, skip_step=3):
    dim_x, dim_y = frame.shape


    if ax is None:
        fig, ax = plt.subplots()

    ax.quiver(np.arange(dim_x)[::skip_step],
              np.arange(dim_y)[::skip_step],
              np.real(frame[::skip_step,::skip_step]).T,
              -np.imag(frame[::skip_step,::skip_step]).T)

    X, Y = np.meshgrid(np.arange(dim_x), np.arange(dim_y), indexing='ij')
    ZR = np.real(frame)
    ZI = np.imag(frame)
    contourR = ax.contour(X,Y, ZR, levels=[0], color='b', label='x = 0')
    contourI = ax.contour(X,Y, ZI, levels=[0], color='g', label='y = 0')
    return ax


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--skip_step", nargs='?', type=int, default=3,
                     help="skipping every x vector for the plot")
    CLI.add_argument("--frame_id", nargs='?', type=int, default=0,
                     help="number of the frame to plot")

    args = CLI.parse_args()
    block = load_neo(args.data)

    block = analogsignals_to_imagesequences(block)

    asig = block.segments[0].analogsignals[0]

    imgseq = block.filter(name='optical_flow', objects="ImageSequence")

    if imgseq:
        imgseq = imgseq[0]
    else:
        raise ValueError("Input does not contain a signal with name " \
                       + "'optical_flow'!")

    crit_point_evt = [evt for evt in block.segments[0].events
                      if evt.name == "Critical Points"]
    if crit_point_evt:
        crit_point_evt = crit_point_evt[0]
    else:
        raise ValueError("Input does not contain an event with name " \
                       + "'Critical Points'!")

    ax = plot_frame(imgseq.as_array()[args.frame_id], skip_step=3)

    start_id = np.argmax(crit_point_evt.times >= asig.times[args.frame_id])
    stop_id = np.argmax(crit_point_evt.times >= asig.times[args.frame_id+1])

    ax.scatter(crit_point_evt.array_annotations['x'][start_id:stop_id],
               crit_point_evt.array_annotations['y'][start_id:stop_id],
               color='r')

    ax.set_title(asig.times[args.frame_id])

    save_plot(args.output)
