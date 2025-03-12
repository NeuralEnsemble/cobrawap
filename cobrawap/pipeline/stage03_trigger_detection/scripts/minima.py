"""
Detect trigger times (i.e., state transition / local wavefronts onsets)
by finding local minima preceding a prominent peak in the channel signals.
"""

import neo
import numpy as np
import quantities as pq
from scipy.signal import find_peaks
import argparse
from pathlib import Path
from utils.io_utils import load_neo, write_neo, save_plot
from utils.neo_utils import remove_annotations, time_slice
from utils.parse import none_or_int, none_or_float, none_or_str
import seaborn as sns
import matplotlib.pyplot as plt


CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--num_interpolation_points", nargs='?', type=int, default=0,
                 help="number of neighboring points to interpolate")
CLI.add_argument("--minima_persistence", nargs='?', type=float, default=0.200,
                 help="minimum time minima (s)")
CLI.add_argument("--maxima_threshold_fraction", nargs='?', type=float, default=0.5,
                 help="amplitude fraction (in range [0,1]) to set the threshold for detecting local maxima")
CLI.add_argument("--maxima_threshold_window", nargs='?', type=none_or_float, default=None,
                 help="time window (s) to set the threshold for detecting local maxima")
CLI.add_argument("--min_peak_distance", nargs='?', type=float, default=0.200,
                 help="minimum distance between peaks (s)")
CLI.add_argument("--img_dir", nargs='?', type=Path,
                 default=None, help="path of figure directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 default='minima_channel0.png',
                 help='example image filename for channel 0')
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="start time (s)")
CLI.add_argument("--plot_tstop", nargs='?', type=none_or_float, default=10,
                 help="stop time (s)")

def filter_minima_order(signal, mins, order=1):
    filtered_mins = np.array([], dtype=int)

    for midx in mins:
        idx = np.arange(midx+1, midx+order, dtype=int)
        idx = idx[idx < len(signal)]

        if (signal[midx] < signal[idx]).all():
            filtered_mins = np.append(filtered_mins, midx)

    return filtered_mins


def moving_threshold(signal, maxima_threshold_window, maxima_threshold_fraction):
    threshold_signal = []

    sampling_rate = signal.sampling_rate.rescale('Hz').magnitude
    duration = float(signal.t_stop.rescale('s').magnitude - signal.t_start.rescale('s').magnitude)
    if maxima_threshold_window is None or maxima_threshold_window > duration:
        maxima_threshold_window = duration
    window_frame = int(maxima_threshold_window*sampling_rate)

    for channel, channel_signal in enumerate(signal.T):
        if np.isnan(channel_signal).any():
            threshold_func = np.full(np.shape(signal)[0], np.nan)
        else:
            # compute a dynamic threshold function through a sliding window
            # on the signal array
            strides = np.lib.stride_tricks.sliding_window_view(channel_signal, window_frame)
            threshold_func = np.min(strides, axis=1) + maxima_threshold_fraction*np.ptp(strides, axis=1)
            # add elements at the beginning
            threshold_func = np.append(np.ones(window_frame//2)*threshold_func[0], threshold_func)
            threshold_func = np.append(threshold_func, np.ones(len(channel_signal)-len(threshold_func))*threshold_func[-1])
        threshold_signal.append(threshold_func)

    threshold_signal = neo.AnalogSignal(np.array(threshold_signal).T, units=signal.units, sampling_rate=signal.sampling_rate)

    return threshold_signal


def detect_minima(asig, threshold_asig, interpolation_points,
                  min_peak_distance, minima_persistence):
    signal = asig.as_array()
    times = asig.times.rescale('s').magnitude
    sampling_rate = asig.sampling_rate.rescale('Hz').magnitude
    threshold = threshold_asig.as_array()

    min_time_idx = np.array([], dtype=int)
    channel_idx  = np.array([], dtype=int)

    minima_order = int(np.max([minima_persistence*sampling_rate, 1]))
    min_distance = np.max([min_peak_distance*sampling_rate, 1])

    for channel, channel_signal in enumerate(signal.T):
        if np.isnan(channel_signal).any(): continue

        peaks, _ = find_peaks(channel_signal, distance=min_distance, height=threshold.T[channel])
        dmins, _ = find_peaks(-channel_signal)#, distance=min_distance)

        mins = filter_minima_order(channel_signal, dmins, order=minima_order)

        clean_mins = np.array([], dtype=int)

        for i, peak in enumerate(peaks):
            distance_to_peak = times[peak] - times[mins]
            distance_to_peak = distance_to_peak[distance_to_peak > 0]
            if distance_to_peak.size:
                trans_idx = np.argmin(distance_to_peak)
                clean_mins = np.append(clean_mins, mins[trans_idx])

        min_time_idx = np.append(min_time_idx, clean_mins)
        channel_idx = np.append(channel_idx, np.ones(len(clean_mins), dtype=int)*channel)

    # compute local minima times.
    if interpolation_points:
        # parabolic fit on the right branch of local minima
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
        minimum_times = min_pos/sampling_rate*pq.s
    else:
        minimum_times = asig.times[min_time_idx]

    idx = np.where(minimum_times >= np.max(asig.times))[0]
    minimum_times[idx] = np.max(asig.times)
    ###################################
    sort_idx = np.argsort(minimum_times)

    # save detected minima as transition
    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='transitions',
                    trigger_detection='minima',
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig)
    evt.annotations.update(asig.annotations)

    return evt


def plot_minima(asig, event, threshold_asig, channel, min_peak_distance):
    signal = asig.as_array().T[channel]
    times = asig.times.rescale('s')
    sampling_rate = asig.sampling_rate.rescale('Hz').magnitude
    threshold = threshold_asig.as_array().T[channel]

    peaks, _ = find_peaks(signal, height=threshold,
                          distance=np.max([min_peak_distance*sampling_rate, 1]))

    # plot figure
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(times, signal, label='signal', color='k')
    ax.plot(times, threshold, label='moving threshold',
            linestyle=':', color='b')

    idx_ch = np.where(event.array_annotations['channels'] == channel)[0]

    ax.plot(times[peaks], signal[peaks], 'x', color='r', label='detected maxima')
    ax.plot(event.times[idx_ch],
            signal[((event.times[idx_ch]-asig.times[0])*sampling_rate).astype(int)],
            'x', color='g', label='selected minima')

    ax.set_title(f'channel {channel}')
    ax.set_xlabel('time [s]')
    ax.legend()

    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    threshold_asig = moving_threshold(asig, args.maxima_threshold_window, args.maxima_threshold_fraction)

    transition_event = detect_minima(asig, threshold_asig,
                                     interpolation_points=args.num_interpolation_points,
                                     min_peak_distance=args.min_peak_distance,
                                     minima_persistence=args.minima_persistence)

    block.segments[0].events.append(transition_event)

    write_neo(args.output, block)

    if args.plot_channels[0] is not None:
        for channel in args.plot_channels:
            plot_minima(asig=time_slice(asig, args.plot_tstart, args.plot_tstop),
                        event=time_slice(transition_event, args.plot_tstart, args.plot_tstop),
                        threshold_asig=time_slice(threshold_asig, args.plot_tstart, args.plot_tstop),
                        channel=int(channel),
                        min_peak_distance=args.min_peak_distance)
            output_path = args.img_dir / args.img_name.replace('_channel0', f'_channel{channel}')
            save_plot(output_path)
