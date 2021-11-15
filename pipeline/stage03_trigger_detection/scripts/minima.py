import neo
import numpy as np
import quantities as pq
from scipy.signal import find_peaks
import argparse
from distutils.util import strtobool
from utils import load_neo, write_neo, remove_annotations, save_plot
from utils import time_slice, none_or_int, none_or_float
import seaborn as sns
import matplotlib.pyplot as plt
import os

def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - 1, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results

def one_side_argrelmin(data, axis=0, order=1, mode='clip'):
    
    results = _boolrelextrema(data, np.less,
                              axis, order, mode)

    return np.nonzero(results)



def detect_minima(asig, interpolation_points, interpolation, maxima_threshold_fraction, maxima_threshold_window, min_peak_distance, minima_persistence):
        
    signal = asig.as_array()
    times = asig.times
    sampling_rate = asig.sampling_rate.rescale('Hz').magnitude
    window_frame = np.int32(maxima_threshold_window*sampling_rate)
    
    min_time_idx = np.array([], dtype=int)
    channel_idx = np.array([], dtype='int32')

    minima_order = np.int32(np.max([minima_persistence*sampling_rate, 1]))
    
    for channel, channel_signal in enumerate(signal.T):
        if np.isnan(channel_signal).any(): continue

        #compute a dynamic treshold function through a sliding window on the signal array
        strides = np.lib.stride_tricks.sliding_window_view(channel_signal, window_frame)
        threshold_func = np.min(strides, axis = 1) + maxima_threshold_fraction*np.ptp(strides, axis = 1)
        #add elements at the beginning
        threshold_func = np.append(np.ones(window_frame//2)*threshold_func[0], threshold_func)
        threshold_func = np.append(threshold_func, np.ones(len(channel_signal) - len(threshold_func))*threshold_func[-1])

        peaks, _ = find_peaks(channel_signal, height=threshold_func, distance=np.max([min_peak_distance*sampling_rate, 1]))
        
        mins_distance, _ = find_peaks(-channel_signal, distance=np.max([min_peak_distance*sampling_rate, 1]))
        mins_persistance = one_side_argrelmin(channel_signal, order = minima_order)
        mins = np.intersect1d(mins_distance, mins_persistance)        

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
        minimum_times = min_pos/sampling_rate*pq.s
    else:
        minimum_times = asig.times[min_time_idx]
    
    ###################################
    sort_idx = np.argsort(minimum_times)
    
    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='Transitions',
                    use_quadtratic_interpolation=interpolation,
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)

    return evt

def plot_minima(asig, event, channel, maxima_threshold_window, maxima_threshold_fraction, min_peak_distance):

    signal = asig.as_array().T[channel]
    sampling_rate = asig.sampling_rate.rescale('Hz').magnitude
    window_frame = np.int32(maxima_threshold_window*sampling_rate) 
    strides = np.lib.stride_tricks.sliding_window_view(signal, window_frame)
    threshold_func = np.min(strides, axis = 1) + maxima_threshold_fraction*np.ptp(strides, axis = 1)
    threshold_func = np.append(threshold_func, np.ones(len(signal) - len(threshold_func))*threshold_func[-1])

    peaks, _ = find_peaks(signal, height=threshold_func, distance=np.max([min_peak_distance*sampling_rate, 1]))  
        
    # plot figure
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(asig.times.rescale('s'), signal, label='signal', color = 'blue', linewidth=1.)
    ax.plot(asig.times.rescale('s'), threshold_func, label='dynamic threshold', color = 'black', linewidth = 0.5)

    idx_ch = np.where(event.array_annotations['channels'] == channel)[0]
    
    ax.plot(asig.times.rescale('s')[peaks], signal[peaks], 'x', color = 'red', label = 'detected maxima') 
    ax.plot(event.times[idx_ch], signal[np.int32(event.times[idx_ch]*sampling_rate)], 'x', color = 'green', label = 'selected minima')

    ax.set_title('Channel {}'.format(channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))
    ax.legend()
    return ax



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
                     help="minimum distance between maxima/minima peaks (s)")
    CLI.add_argument("--minima_persistence", nargs='?', type=float, default=0.200,
                     help="minimum time minima (s)")
    CLI.add_argument("--maxima_threshold_fraction", nargs='?', type=float, default=0.,
                     help="amplitude fraction to set the threshold detecting local maxima")
    CLI.add_argument("--maxima_threshold_window", nargs='?', type=int, default=None,
                     help="time window to use to set the threshold detecting local maxima")
    
    CLI.add_argument("--img_dir", nargs='?', type=str,
                     default='None', help="path of figure directory")
    CLI.add_argument("--img_name", nargs='?', type=str, default='minima_channel0.png',
                     help='example image filename for channel 0')
    CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                     help="list of channels to plot")
    CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0.,
                     help="start time in seconds")
    CLI.add_argument("--plot_tstop",  nargs='?', type=none_or_float, default=40.,
                     help="stop time in seconds")


    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig,
                                     interpolation_points=args.num_interpolation_points,
                                     interpolation=args.use_quadtratic_interpolation,
                                     maxima_threshold_fraction=args.maxima_threshold_fraction,
                                     maxima_threshold_window=args.maxima_threshold_window,
                                     min_peak_distance=args.min_peak_distance,
                                     minima_persistence=args.minima_persistence)
    
   
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)

    if args.plot_channels[0] is not None:
        for channel in args.plot_channels:
            plot_minima(asig=time_slice(asig, args.plot_tstart, args.plot_tstop),
                        event=time_slice(transition_event, args.plot_tstart, args.plot_tstop),
                        channel=int(channel),
                        maxima_threshold_window = args.maxima_threshold_window,
                        maxima_threshold_fraction = args.maxima_threshold_fraction, 
                        min_peak_distance = args.min_peak_distance)
            output_path = os.path.join(args.img_dir,
                                       args.img_name.replace('_channel0', f'_channel{channel}'))
            save_plot(output_path)
