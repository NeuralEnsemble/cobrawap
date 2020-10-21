import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin
import argparse
from utils import load_neo, write_neo, remove_annotations


def detect_minima(asig, order, interpolation_points, interpolation):
    signal = asig.as_array()
    sampling_time = asig.times[1] - asig.times[0]

    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)
    
    if interpolation:

        fitted_idx_times = np.zeros([len(t_idx)])
        
        start_arr = t_idx - int(interpolation_points/2)
        start_arr = np.where(start_arr > 0, start_arr, 0)
        stop_arr = start_arr + int(interpolation_points)
        start_arr = np.where(stop_arr < len(signal), start_arr, len(signal)-interpolation_points-1)
        stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal)-1)
        
        signal_arr = np.empty((interpolation_points,len(start_arr)))
        signal_arr[:] = np.nan
        
        for i, (start, stop, channel_i) in enumerate(zip(start_arr, stop_arr, channel_idx)):
            signal_arr[:,i] = signal[start:stop, channel_i]

        X_temp = range(0, interpolation_points)
        params = np.polyfit(X_temp, signal_arr, 2)
        
        min_pos = -params[1,:] / (2*params[0,:]) + start_arr
        min_pos = np.where(min_pos > 0, min_pos, 0)
        minimum_times = min_pos * sampling_time

        
    else:
        minimum_times = asig.times[t_idx]
    
    sort_idx = np.argsort(minimum_times)

    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='Transitions',
                    minima_order=order,
                    use_quadtratic_interpolation=interpolation,
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx[sort_idx]},
                    )

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
    CLI.add_argument("--use_quadtratic_interpolation", nargs='?', type=bool, default=True,
                     help="wether use interpolation or not")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    transition_event = detect_minima(asig, args.order, args.num_interpolation_points, args.use_quadtratic_interpolation)
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)

