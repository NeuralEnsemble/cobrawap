"""
Detect and characterize critical points in the optical flow vector field at each
time point.
"""

import argparse
from pathlib import Path
import numpy as np
import neo
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from utils.io_utils import load_neo, write_neo, save_plot
from utils.neo_utils import analogsignal_to_imagesequence

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")

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
    extend = np.array([], dtype=int)
    winding_number = np.array([], dtype=int)
    # winding_number = np.array([], dtype=float) # ToDo

    for i in range(dim_t):
        # ToDo: parallelize
        Y, X = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
        ZR = np.real(frames[i])
        ZI = np.imag(frames[i])
        contourR = plt.contour(X, Y, ZR, levels=[0])
        contourI = plt.contour(X, Y, ZI, levels=[0])

        for xy in get_line_intersections(contourR, contourI):
            x = np.append(x, xy[0])
            y = np.append(y, xy[1])
            frame_ids = np.append(frame_ids, i)

            J = jacobian(xy, ZR, ZI)
            trace_i = np.trace(J)
            det_i = np.linalg.det(J)
            labels = np.append(labels, classify_critical_point(det=det_i,
                                                               trace=trace_i))
            trace = np.append(trace, trace_i)
            det = np.append(det, det_i)

            extend_i, winding_number_i = calc_winding_number(xy, frames[i])
            extend = np.append(extend, extend_i)
            winding_number = np.append(winding_number, winding_number_i)

    evt = neo.Event(name='critical_points',
                    times=times[frame_ids],
                    labels=labels)
    evt.array_annotations.update({'x':x, 'y':y,
                                  'trace':trace, 'det':det,
                                  'extend':extend,
                                  'winding_number':winding_number})
    return evt


def jacobian(xy, fA, fB):
    # dA/dy  dA/dx
    # dB/dy  dB/dx
    x, y = int(np.round(xy[0])), int(np.round(xy[1]))
    J = np.zeros((2,2))
    dim_y, dim_x = fA.shape
    if y+1 < dim_y:
        J[0,0] = fA[y+1, x] - fA[y, x]
        J[1,0] = fB[y+1, x] - fB[y, x]
    else:
        J[0,0] = fA[y, x] - fA[y-1, x]
        J[1,0] = fB[y, x] - fB[y-1, x]
    if x+1 < dim_x:
        J[0,1] = fA[y, x+1] - fA[y, x]
        J[1,1] = fB[y, x+1] - fB[y, x]
    else:
        J[0,1] = fA[y, x] - fA[y, x-1]
        J[1,1] = fB[y, x] - fB[y, x-1]
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


def calc_winding_number(xy, frame):
    px, py = int(np.round(xy[0])), int(np.round(xy[1]))
    dim_y, dim_x = frame.shape
    max_r = np.ceil(min([dim_x, dim_y])/2)
    prev_winding_number = 0
    for r in range(1, int(max_r)):
        # is circle intersecting with the image border?
        if px + r >= dim_x or px - r <= 0:
            break
        elif py + r >= dim_y or py - r <= 0:
            break
        else:
            Y, X = np.indices((dim_y, dim_x))
            circle = np.abs(np.hypot(py-Y,px-X) - r) < 0.5

        # extracting counter-clock-wise indices of circle with radius r
        ccw = np.array([], dtype=[('x',int), ('y',int)])
        # x0     x1     y0    y1     sx   sy
        # 0  0   xp xp  0 py  py -1  + -  - -
        # xp xp  -1 -1  0 py  py -1  + -  + +
        for x0, x1, y0, y1 in zip([0 , px, px, 0], # Quadrants TL, BL, BR, TR
                                  [px, -1, -1, px],
                                  [0 ,  0, py, py],
                                  [py, py, -1, -1]):
            circle_yi, circle_xi = np.where(circle[y0:y1, x0:x1])
            circle_xi += x0
            circle_yi += y0
            circle_idx = np.array([(-np.sign(x1)*yi, np.sign(y1)*xi)
                                    for xi,yi in zip(circle_xi, circle_yi)],
                                  dtype=[('x', int), ('y', int)])
            ccw = np.append(ccw, np.sort(circle_idx, order=['y', 'x']))

        # sum differences of subsequent vector angles around circle
        circle_values = frame[np.abs(ccw['y']), np.abs(ccw['x'])]
        circle_angles = np.angle(circle_values) + np.pi
        circle_angles = np.append(circle_angles, circle_angles[0])

        # ToDo: solve more pythonic
        sum_angles = 0
        for i, phi in enumerate(circle_angles[:-1]):
            dphi = circle_angles[i+1] - phi
            if dphi > np.pi:
                dphi -= 2*np.pi
            elif dphi < -np.pi:
                dphi += 2*np.pi
            sum_angles += dphi
        winding_number = sum_angles/(2*np.pi)

        winding_number = np.round(winding_number)
        if r != 1 and prev_winding_number != winding_number:
            break
        else:
            prev_winding_number = winding_number
    return r-1, prev_winding_number


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
    args, unknown = CLI.parse_known_args()
    block = load_neo(args.data)

    asig = block.filter(name='optical_flow', objects="AnalogSignal")[0]
    imgseq = analogsignal_to_imagesequence(asig)

    crit_point_evt = detect_critical_points(imgseq,
                                    block.segments[0].analogsignals[0].times)

    block.segments[0].events.append(crit_point_evt)

    write_neo(args.output, block)
