import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin, argrelmax, find_peaks
import argparse
from distutils.util import strtobool
from utils import load_neo, write_neo, remove_annotations


def detect_minima(asig, order, interpolation_points, interpolation, threshold_fraction,  Min_Peak_Distance):
        
    signal = asig.as_array()
    times = asig.times
    sampling_time = asig.times[1] - asig.times[0]
    min_idx, channel_idx_minima = argrelmin(signal, order=order, axis=0)

    amplitude_span = np.max(signal, axis = 0) - np.min(signal, axis = 0)
    threshold = np.min(signal, axis = 0) + threshold_fraction*(amplitude_span)
    
    
    min_time_idx = []
    channel_idx = []
    for ch in range(len(signal[0])):
        peaks, _ = find_peaks(signal.T[ch], height=threshold[ch], distance = np.int32(Min_Peak_Distance/sampling_time))#, prominence=prominence)
        mins = min_idx[np.where(channel_idx_minima == ch)[0]]

        clean_mins = np.array([], dtype=int)
        for i, peak in enumerate(peaks):
            distance_to_peak = times[peak] - times[mins]
            distance_to_peak = distance_to_peak[distance_to_peak > 0]
            if distance_to_peak.size:
                trans_idx = np.argmin(distance_to_peak)
                clean_mins = np.append(clean_mins, mins[trans_idx])

        min_time_idx.extend(clean_mins)
        channel_idx.extend(list(np.ones(len(clean_mins))*ch))
        
    
    # compute local minima times.
    if interpolation:
        # parabolic fit around the local minima
        fitted_idx_times = np.zeros([len(min_time_idx)])
        start_arr = min_time_idx - 1 #int(interpolation_points/2)
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
        minimum_times = min_pos * sampling_time
        minimum_value = params[0,:]*( -params[1,:] / (2*params[0,:]) )**2 + params[1,:]*( -params[1,:] / (2*params[0,:]) ) + params[2,:]

        minimum_times[np.where(minimum_times > asig.t_stop)[0]] = asig.t_stop
    else:
        minimum_times = asig.times[min_time_idx]
    
    ###################################
    sort_idx = np.argsort(minimum_times)
    channel_idx = np.int32(channel_idx)
    
    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='Transitions',
                    minima_order=order,
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
    CLI.add_argument("--order", nargs='?', type=int, default=3,
                     help="number of neighbouring points to compare")
    CLI.add_argument("--num_interpolation_points", nargs='?', type=int, default=5,
                     help="number of neighbouring points to interpolate")
    CLI.add_argument("--use_quadtratic_interpolation", nargs='?', type=strtobool, default=False,
                     help="wether use interpolation or not")
    CLI.add_argument("--min_peak_distance", nargs='?', type=float, default=0.200,
                     help="minimum distance between peacks (s)")
    CLI.add_argument("--threshold_fraction", nargs='?', type=float, default=0.,
                     help="amplitude fraction to set the threshold detecting local maxima")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig,
                                     order=args.order,
                                     interpolation_points=args.num_interpolation_points, interpolation=args.use_quadtratic_interpolation, threshold_fraction = args.threshold_fraction,  Min_Peak_Distance = args.min_peak_distance)
    
   
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)
