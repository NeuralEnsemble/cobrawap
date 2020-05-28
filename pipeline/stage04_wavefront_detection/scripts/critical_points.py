import argparse
import numpy as np
import neo
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from utils import load_neo, write_neo, none_or_str, save_plot, \
                  ImageSequence2AnalogSignal, AnalogSignal2ImageSequence


def detect_critical_points(imgseq, times):
    frames = imgseq.as_array()
    if frames.dtype != np.complex128:
        raise ValueError("Vector field values must be complex numbers!")

    dim_t, dim_x, dim_y = frames.shape

    frame_ids = np.array([], dtype=int)
    labels = np.array([], dtype=str)
    x = np.array([], dtype=float)
    y = np.array([], dtype=float)
    trace = np.array([], dtype=float)
    det = np.array([], dtype=float)
    # winding_number = np.array([], dtype=float) # ToDo

    for i in range(dim_t):
        # ToDo: parallelize
        X, Y = np.meshgrid(np.arange(dim_x), np.arange(dim_y))
        ZR = np.real(frames[i])
        ZI = np.imag(frames[i])
        contourR = plt.contour(X,Y, ZR, levels=[0])
        contourI = plt.contour(X,Y, ZI, levels=[0])

        for xy in  get_line_intersections(contourR, contourI):
            x = np.append(x, xy[0])
            y = np.append(y, xy[1])
            frame_ids = np.append(frame_ids, i)

            J = jacobian(xy, ZR.T, ZI.T)
            trace_i = np.trace(J)
            det_i = np.linalg.det(J)
            labels = np.append(labels, classify_critical_point(det=det_i,
                                                               trace=trace_i))
            trace = np.append(trace, trace_i)
            det = np.append(det, det_i)

    evt = neo.Event(name='Critical Points',
                    times=times[frame_ids],
                    labels=labels)
    evt.array_annotate(x=x, y=y, trace=trace, det=det)
    return evt


def jacobian(xy, fA, fB):
    # dA/dx  dA/dy
    # dB/dx  dB/dy
    x, y = int(xy[0]), int(xy[1])
    J = np.zeros((2,2))
    J[0,0] = fA[x+1, y]   - fA[x, y]
    J[0,1] = fA[x  , y+1] - fA[x, y]
    J[1,0] = fB[x+1, y]   - fB[x, y]
    J[1,1] = fB[x  , y+1] - fB[x, y]
    return J


def classify_critical_point(det, trace):
    if det > 0:
        if trace**2 > 4*det:
            if trace > 0 :
                return 'node stable'
            else:
                return 'node unstable'
        else:
            if trace > 0:
                return 'focus stable'
            else:
                return 'focus unstable'
    else:
        return 'saddle'


def get_line_intersections(contourA, contourB):
    points = []
    for path in contourA.collections[0].get_paths():
        if len(path.vertices) > 1:
            lineA = LineString(path.vertices)
        else:
            lineA = None

        for path in contourB.collections[0].get_paths():
            if len(path.vertices) > 1 and lineA is not None:
                lineB = LineString(path.vertices)

                intersection = lineA.intersection(lineB)

                if not intersection.is_empty:
                    if intersection.type == 'MultiPoint':
                        points += [(i.x, i.y) for i in intersection]
                    elif intersection.type == 'Point':
                        points += [(intersection.x, intersection.y)]
    return points


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")

    args = CLI.parse_args()
    block = load_neo(args.data)

    block = AnalogSignal2ImageSequence(block)

    imgseq = [im for im in block.segments[0].imagesequences
                        if im.name == "Optical Flow"]
    if imgseq:
        imgseq = imgseq[0]
    else:
        raise ValueError("Input does not contain a signal with name " \
                       + "'Optical Flow'!")


    crit_point_evt = detect_critical_points(imgseq,
                                    block.segments[0].analogsignals[0].times)

    block.segments[0].events.append(crit_point_evt)

    write_neo(args.output, block)
