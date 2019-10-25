# Calculating the velocities of wavefronts
# Assumptions: wavefronts are planar, and have constant velocity
import neo
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import elephant as el
import scipy
import quantities as pq

def up_pixel_per_frame(up_trains, times):
    up_coords = [[] for t in times]
    for up_train in up_trains:
        for up_time in up_train:
            t_idx = np.where(times >= up_time)[0][0]
            up_coords[t_idx].append(up_train.annotations['coordinates'])
    return up_coords

def detect_waves(up_coords, min_pixel, max_fiterror, max_rotation):
    waves = [None for t in up_coords]
    for i, wave in enumerate(waves):
        # select wavefront with a minimum number of participating pixels
        if len(up_coords[i]) > min_pixel:
            pixels = np.array(up_coords[i]).T
            # define continous wavefront, assuming it is linear
            slope, intercept, _, _, stderr = scipy.stats.linregress(pixels[1], pixels[0])
            # check that the direction of the wavefront doesn't change more than a certain angle
            if i and waves[i-1] is not None:
                prev_slope = waves[i-1][0]
                angle = np.abs(np.arctan((slope-prev_slope)/(1+slope*prev_slope)))
            else:
                angle = 0
            # exclude diffuse up transition which don't form a linear wave front
            if stderr < max_fiterror and angle < max_rotation:
                waves[i] = (slope, intercept)
    return waves

def remove_isolated_waves(waves, min_frame_num):
    count = 0
    for i, wave in enumerate(waves):
        if wave is not None and i < len(waves)-1:
            count += 1
        elif count:
            if count < min_frame_num:
                for c in range(count+1):
                    waves[i-c] = None
            count = 0
    return waves

def wave_to_wave_distance(waves, contour):
    def find_intersections(function, points, **func_kwargs):
            p_above_f = np.array([point[1] >= function(point[0], **func_kwargs) for point in points])
            idx = np.where(p_above_f[:-1] != p_above_f[1:])[0]
            x_itsct = [points[idx[0]][0], points[idx[-1]][0]]
            return np.array([[x, function(x, **func_kwargs)] for x in x_itsct])

    wave_front = lambda x, i: waves[i][1] + x*waves[i][0]

    # intersections of planar wavefront with contour
    isec = [None if wave is None
            else find_intersections(wave_front, contour, i=i)
            for i, wave in enumerate(waves)]

    def P1P2_to_P3(p1, p2, p3):
        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

    # average distance between sucessive wavefronts
    displacements = np.zeros(len(isec))
    for i in range(len(isec)):
        if isec[i] is not None and isec[i+1] is not None:
            dists = [np.abs(P1P2_to_P3(isec[i][0], isec[i][1], isec[i+1][j]))
                     for j in range(2)]
            displacements[i] = np.mean(dists)
    return displacements


def calc_velocity(displacements, sampling_rate, pixel_size):
    indices = np.where(displacements)[0]
    idx_groups = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
    dist_groups = [displacements[idx] for idx in idx_groups]
    cumdist_groups = [np.cumsum(group) for group in dist_groups]
    velocity = np.zeros((2,len(dist_groups)))

    for i, dist in enumerate(cumdist_groups):
        dist = np.append(0, dist) * pixel_size.magnitude
        x = np.arange(len(dist)) / sampling_rate.magnitude
        slope, intercept, _, _, stderr = scipy.stats.linregress(x, dist)
        velocity[0][i] = slope
        velocity[1][i] = stderr

    return velocity



if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--waves", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--contour", nargs='?', type=str)
    CLI.add_argument("--min_pixelfrac", nargs='?', type=float)
    CLI.add_argument("--max_fiterror", nargs='?', type=float)
    CLI.add_argument("--max_rotation", nargs='?', type=float)
    CLI.add_argument("--min_frames", nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.waves) as io:
        seg = io.read_block().segments[0]
        up_trains = seg.spiketrains
        images = seg.analogsignals[0]

    # for every frame, create list of pixels which show a up transition
    up_coords = up_pixel_per_frame(up_trains, images.times)

    # linear fit through UP pixels;
    # select for minimum number of pixels, maximal fit error,
    # and maximal rotation of the wavefront direction
    pixel_num = (~np.isnan(images[0])).sum()
    waves = detect_waves(up_coords,
                         min_pixel = args.min_pixelfrac*pixel_num,
                         max_fiterror = args.max_fiterror,
                         max_rotation = args.max_rotation)

    waves = remove_isolated_waves(waves,
                                  min_frame_num = args.min_frames)

    # calculate distances between wavefronts(t) within contour
    contour = np.load(args.contour)
    displacements = wave_to_wave_distance(waves, contour)

    # calculate velocity of each wave speparately
    velocity = calc_velocity(displacements,
                            sampling_rate=images.sampling_rate,
                            pixel_size=images.annotations['pixel_size']*pq.mm)
    np.save(args.output, velocity)
