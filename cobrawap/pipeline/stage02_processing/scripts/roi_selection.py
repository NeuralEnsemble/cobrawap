"""
Select a region of interest (ROI) by thresholding the intensity signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import shapely.geometry as geo
import argparse
from pathlib import Path
import neo
import os
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_path, str_to_bool
from utils.neo_utils import analogsignal_to_imagesequence, imagesequence_to_analogsignal

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path,
                 help="path of output image", default=None)
CLI.add_argument("--intensity_threshold", nargs='?', type=float,
                 help="threshold for mask [0,1]", default=0.5)
CLI.add_argument("--crop_to_selection", nargs='?', type=str_to_bool,
                 help="discard frame outside of ROI", default=True)

def calculate_contour(img, contour_limit):
    # Computing the contour lines...
    vmax = np.max(img)
    vmin = np.min(img)
    limit = vmin + contour_limit * (vmax-vmin)
    contour_fake = measure.find_contours(img, limit)
    # Select the largest contour lines
    max_index = np.argmax([len(c) for c in contour_fake])

    contour = contour_fake[max_index]
    contour = np.flip(contour, 1) # x,y

    def border_intercepts(contour):
        intercept_left = False
        intercept_right = False
        intercept_top = False
        intercept_bot = False

        if not np.all(contour[:,0]):
            intercept_left = True
        if not np.all(contour[:,1]):
            intercept_bot = True
        if not np.all(contour[:,0]-len(img[0])+1):
            intercept_right = True
        if not np.all(contour[:,1]-len(img[1])+1):
            intercept_top = True
        return {'left': intercept_left, 'right': intercept_right,
                'top': intercept_top, 'bottom' : intercept_bot}

    contour_intercepts = border_intercepts(contour)
    if sum(contour_intercepts.values()) > 1:
        includes_corner = True
        del contour_fake[max_index]

        # include secondary connecting contour
        connected = False
        for snd_contour in contour_fake:
            snd_contour = np.flip(snd_contour, 1) # x,y
            if border_intercepts(snd_contour) == contour_intercepts:
                contour = np.append(contour, snd_contour, axis=0)
                connected = True
        # or include corner
        if not connected:
            snd_contour = []
            if contour_intercepts['left'] and contour_intercepts['bottom']:
                snd_contour += [0,0]
            elif contour_intercepts['bottom'] and contour_intercepts['right']:
                snd_contour += [len(img[0])-1,0]
            elif contour_intercepts['left'] and contour_intercepts['bottom']:
                snd_contour += [0,len(img[1]-1)]
            elif contour_intercepts['top'] and contour_intercepts['right']:
                snd_contour += [len(img[0])-1,len(img[1])-1]
            else:
                raise ValueError('The contour is to large, and can not be determined unambigously!')
    else:
        includes_corner = False

    print('Contour includes corner = ', includes_corner)
    return contour


def close_contour(contour, num):
    dx = contour[-1][0] - contour[0][0]
    dy = contour[-1][1] - contour[0][1]
    avg_d = np.mean(np.append(np.abs(np.diff(contour[:,0])),
                              np.abs(np.diff(contour[:,1]))))
    num = int(np.sqrt(dx**2 + dy**2) / avg_d)
    def connect_coords(idx, num):
        connect = [contour[-1][idx], contour[0][idx]]
        return np.linspace(connect[0], connect[1], num)
    contour_connect = np.array([connect_coords(i, num)
                                for i in range(2)]).T
    return np.concatenate((contour, contour_connect))


def contour2mask(contour, dim_x, dim_y):
    mask = np.zeros((dim_y, dim_x), dtype=bool)
    polygon = geo.polygon.Polygon(contour)
    for x in range(dim_x):
        for y in range(dim_y):
            point = geo.Point(x,y)
            if polygon.contains(point):
                mask[y,x] = 1
    return mask


def crop_to_selection(frames):
    frame = frames[0]
    y, x = np.where(np.isfinite(frame))
    x0, x1 = np.min(x), np.max(x)
    y0, y1 = np.min(y), np.max(y)
    return frames[:, y0:y1+1, x0:x1+1]


def plot_roi(img, contour):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray, origin='lower')
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(contour[:,0], contour[:,1], linewidth=2)
    plt.draw()
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    imgseq = analogsignal_to_imagesequence(asig)

    # get average image
    imgseq_array = imgseq.as_array()
    dim_t, dim_y, dim_x = imgseq_array.shape
    avg_img = np.mean(imgseq_array, axis=0)

    # calculate mask
    contour = calculate_contour(img=avg_img,
                                contour_limit=args.intensity_threshold)
    contour = close_contour(contour, num=100)
    mask = contour2mask(contour=contour,
                        dim_x=dim_x,
                        dim_y=dim_y)

    # apply mask
    imgseq_array[:, np.bitwise_not(mask)] = np.nan
    if args.crop_to_selection:
        imgseq_array = crop_to_selection(imgseq_array)

    # replace analogsingal
    tmp_blk = neo.Block()
    tmp_seg = neo.Segment()
    tmp_blk.segments.append(tmp_seg)
    tmp_imgseq = imgseq.duplicate_with_new_data(imgseq_array)
    new_asig = imagesequence_to_analogsignal(tmp_imgseq)

    # save data and figure
    new_asig.description += "Border regions with mean intensity below "\
                         + "{args.intensity_threshold} were discarded. "\
                         + "({})".format(os.path.basename(__file__))
    block.segments[0].analogsignals = [new_asig]
    # TODO: What about other analogsignals, e.g. hemodynamics_subtraction data?
    #       Do they need to be cropped as well?

    plot_roi(avg_img, contour)
    if args.output_img is not None:
        save_plot(args.output_img)

    write_neo(args.output, block)
