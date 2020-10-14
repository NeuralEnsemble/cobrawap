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

    if int(interpolation) == 0:

        fitted_idx_times = np.zeros([len(t_idx)])
        
        start_arr = np.array(t_idx) - int(interpolation_points/2)
        start_arr = np.where(start_arr > 0, start_arr, 0)
        stop_arr = start_arr + int(interpolation_points)
        stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal))
        
        for start, stop, channel_i, i in zip(start_arr, stop_arr, channel_idx, range(len(start_arr))):
            X = list(range(start, stop))
            params = np.polyfit(X, signal[start:stop, channel_i], 2)
            minimum = -(1.*params[1])/(2.*params[0])
            fitted_idx_times[i] = minimum*sampling_time
            
        fitted_idx_times = np.where(fitted_idx_times > 0, fitted_idx_times, 0)
        sort_idx = np.argsort(fitted_idx_times)

    else:
        fitted_idx_times = np.asarray(asig.times[t_idx]*sampling_time)
        sort_idx = np.argsort(fitted_idx_times)
        
    
    evt = neo.Event(times=fitted_idx_times[sort_idx]*asig.times.units,
                    labels=['UP'] * len(fitted_idx_times),
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
    CLI.add_argument("--use_quadtratic_interpolation", nargs='?', type=bool, default=0,
                     help="wether use interpolation or not")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    transition_event = detect_minima(asig, args.order, args.num_interpolation_points, args.use_quadtratic_interpolation)
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)

