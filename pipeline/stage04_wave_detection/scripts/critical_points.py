import argparse
import numpy as np
import neo
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from utils.io import load_neo, write_neo, save_plot
from utils.neo import analogsignals_to_imagesequences


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
        X, Y = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='xy')
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

    evt = neo.Event(name='Critical Points',
                    times=times[frame_ids],
                    labels=labels)
    evt.array_annotations.update({'x':x, 'y':y, 'trace':trace, 'det':det,
                                  'extend':extend, 'winding_number':winding_number})
    return evt


def jacobian(xy, fA, fB):
    # dA/dx  dA/dy
    # dB/dx  dB/dy
    x, y = int(np.round(xy[0])), int(np.round(xy[1]))
    J = np.zeros((2,2))
    dim_x, dim_y = fA.shape
    if x+1 < dim_x:
        J[0,0] = fA[x+1, y] - fA[x, y]
        J[1,0] = fB[x+1, y] - fB[x, y]
    else:
        J[0,0] = fA[x, y] - fA[x-1, y]
        J[1,0] = fB[x, y] - fB[x-1, y]
    if y+1 < dim_y:
        J[0,1] = fA[x, y+1] - fA[x, y]
        J[1,1] = fB[x, y+1] - fB[x, y]
    else:
        J[0,1] = fA[x, y] - fA[x, y-1]
        J[1,1] = fB[x, y] - fB[x, y-1]
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
    dim_x, dim_y = frame.shape
    max_r = np.ceil(min([dim_x, dim_y])/2)
    prev_winding_number = 0
    for r in range(1, int(max_r)):
        # is circle intersecting with the image border?
        if px + r >= dim_x or px - r <= 0:
            break
        elif py + r >= dim_y or py - r <= 0:
            break
        else:
            X, Y = np.indices((dim_x, dim_y))
            circle = np.abs(np.hypot(px-X, py-Y) - r) < 0.5

        # extracting counter-clock-wise indices of circle with radius r
        ccw = np.array([], dtype=[('x',int), ('y',int)])
        # x0     x1     y0    y1     sx   sy
        # 0  0   xp xp  0 py  py -1  + -  - -
        # xp xp  -1 -1  0 py  py -1  + -  + +
        for x0, x1, y0, y1 in zip([0 , px, px, 0], # Quadrants TL, BL, BR, TR
                                  [px, -1, -1, px],
                                  [0 ,  0, py, py],
                                  [py, py, -1, -1]):
            circle_xi, circle_yi = np.where(circle[x0:x1, y0:y1])
            circle_xi += x0
            circle_yi += y0
            circle_idx = np.array([(np.sign(y1)*xi, -np.sign(x1)*yi)
                                    for xi,yi in zip(circle_xi, circle_yi)],
                                  dtype=[('x', int), ('y', int)])
            ccw = np.append(ccw, np.sort(circle_idx, order=['x', 'y']))

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
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")

    args = CLI.parse_args()
    block = load_neo(args.data)

    block = analogsignals_to_imagesequences(block)

    imgseq = block.filter(name='optical_flow', objects="ImageSequence")

    if imgseq:
        imgseq = imgseq[0]
    else:
        raise ValueError("Input does not contain a signal with name " \
                       + "'optical_flow'!")


    crit_point_evt = detect_critical_points(imgseq,
                                    block.segments[0].analogsignals[0].times)

    block.segments[0].events.append(crit_point_evt)

    write_neo(args.output, block)
