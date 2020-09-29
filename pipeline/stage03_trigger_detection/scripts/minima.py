import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin
import argparse
from utils import load_neo, write_neo, remove_annotations


def detect_minima(asig, order, interpolation_points, sampling_time, interpolation):
    signal = asig.as_array()
    
    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)

    if int(interpolation) == 0:
        fitted_idx = []
        fitted_idx_times = []

        for i in range(0,len(t_idx)):
        
            if t_idx[i] - int((interpolation_points-1)/2) >= 0 and t_idx[i] + int((interpolation_points-1)/2)+1 <= len(signal[:,0]):
                start = t_idx[i] - int((interpolation_points-1)/2)
                end =  t_idx[i] + int((interpolation_points-1)/2)+1

                X = [i for i in range(start,end)]

                params = np.polyfit(X, signal[start:end, channel_idx[i]], 2)

                temp = [params[0]*x**2 + params[1]*x + params[2] for x in X]
                minimum = -(1.*params[1])/(2.*params[0])
                if minimum > 0:
                    fitted_idx.append(minimum)
                    fitted_idx_times.append(1.*minimum/sampling_time)
            
        fitted_idx_times = np.asarray(fitted_idx_times)
        sort_idx = np.argsort(fitted_idx_times)
    else:
        fitted_idx_times = np.asarray(float(t_idx)/sampling_time)
        sort_idx = np.argsort(fitted_idx_times)
        
    
    evt = neo.Event(times=fitted_idx_times[sort_idx]*pq.s,
                    labels=['UP'] * len(fitted_idx_times),
                    name='Transitions',
                    minima_order=order,
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
    CLI.add_argument("--interpolation_points", nargs='?', type=int, default=5,
                     help="number of neighbouring points to interpolate")
    CLI.add_argument("--sampling_time", nargs='?', type=int, default=25,
                     help="sampling time [Hz]")
    CLI.add_argument("--interpolation", nargs='?', type=int, default=0,
                     help="wether use interpolation or not")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    transition_event = detect_minima(asig, args.order, args.interpolation_points, args.sampling_time, args.interpolation)
    
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)
