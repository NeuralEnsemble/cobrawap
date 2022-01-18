import neo
import numpy as np
import quantities as pq
from scipy.signal import find_peaks
import argparse
from distutils.util import strtobool
from utils.io import load_neo, write_neo
from utils.neo import remove_annotations


def detect_minima(asig, interpolation_points, interpolation, min_peak_distance
                  minima_threshold_fraction, maxima_threshold_fraction):

    signal = asig.as_array()
    times = asig.times
    sampling_rate = asig.sampling_rate.rescale('Hz').magnitude

    amplitude_span = np.max(signal, axis=0) - np.min(signal, axis=0)
    maxima_threshold = np.min(signal, axis=0) + maxima_threshold_fraction*(amplitude_span)
    minima_threshold = np.max(-signal, axis=0) - minima_threshold_fraction*(amplitude_span)


    min_time_idx = np.array([], dtype=int)
    channel_idx = np.array([], dtype='int32')

    for channel, channel_signal in enumerate(signal.T):
        if np.isnan(channel_signal).any(): continue
        peaks, _ = find_peaks(channel_signal,
                              height=maxima_threshold[channel],
                              distance=np.max([min_peak_distance*sampling_rate, 1]))
        mins, _ = find_peaks(-channel_signal,
                             height=minima_threshold[channel],
                             distance=np.max([min_peak_distance*sampling_rate, 1]))

        clean_mins = np.array([], dtype=int)
        for i, peak in enumerate(peaks):
            distance_to_peak = times[peak] - times[mins]
            distance_to_peak = distance_to_peak[distance_to_peak > 0]
            if distance_to_peak.size:
                trans_idx = np.argmin(distance_to_peak)
                clean_mins = np.append(clean_mins, mins[trans_idx])

        min_time_idx = np.append(min_time_idx, clean_mins)
        channel_idx = np.append(channel_idx, np.ones(len(clean_mins), dtype='int32')*channel)

    # compute local minima times.
    if interpolation:
        # parabolic fit on the right branch of local minima
        fitted_idx_times = np.zeros([len(min_time_idx)])
        start_arr = min_time_idx - 1
        start_arr = np.where(start_arr > 0, start_arr, 0)
        stop_arr = start_arr + int(interpolation_points)

        start_arr = np.where(stop_arr < len(signal), start_arr, len(signal)-interpolation_points-1)
        stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal)-1)

        signal_arr = np.empty((interpolation_points, len(start_arr)))
        signal_arr.fill(np.nan)

        for i, (start, stop, channel_i) in enumerate(zip(start_arr, stop_arr, channel_idx)):
            signal_arr[:,i] = signal[start:stop, channel_i]

        X_temp = range(0, interpolation_points)
        params = np.polyfit(X_temp, signal_arr, 2)

        min_pos = -params[1,:] / (2*params[0,:]) + start_arr
        min_pos = np.where(min_pos > 0, min_pos, 0)
        minimum_times = min_pos/sampling_rate
    else:
        minimum_times = asig.times[min_time_idx]

    ###################################
    sort_idx = np.argsort(minimum_times)

    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='transitions',
                    use_quadtratic_interpolation=interpolation,
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)

    return evt



if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--num_interpolation_points", nargs='?', type=int, default=5,
                     help="number of neighbouring points to interpolate")
    CLI.add_argument("--use_quadtratic_interpolation", nargs='?', type=strtobool, default=False,
                     help="wether use interpolation or not")
    CLI.add_argument("--min_peak_distance", nargs='?', type=float, default=0.200,
                     help="minimum distance between peaks (s)")
    CLI.add_argument("--minima_threshold_fraction", nargs='?', type=float, default=0.,
                     help="amplitude fraction to set the threshold detecting local minima")
    CLI.add_argument("--maxima_threshold_fraction", nargs='?', type=float, default=0.,
                     help="amplitude fraction to set the threshold detecting local maxima")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig,
                                     interpolation_points=args.num_interpolation_points,
                                     interpolation=args.use_quadtratic_interpolation,
                                     minima_threshold_fraction=args.minima_threshold_fraction,
                                     maxima_threshold_fraction=args.maxima_threshold_fraction,
                                     min_peak_distance=args.min_peak_distance)


    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)
