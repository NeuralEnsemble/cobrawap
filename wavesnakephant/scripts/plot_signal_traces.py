import neo
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import argparse
import quantities as pq
from load_and_transform_to_neo import load_segment


def plot_signal_traces(segment, t_start, t_stop, scaling):
    fig, ax = plt.subplots()
    asig = segment.analogsignals[0]
    num_sampling_points = (asig.t_stop.rescale('s') - asig.t_start.rescale('s')) \
                          * asig.sampling_rate.rescale('1/s')
    sampling_times = np.linspace(asig.t_start, asig.t_stop, num_sampling_points)
    idx = [int(t * asig.sampling_rate.rescale('1/s').magnitude)
           for t in [t_start.rescale('s').magnitude, t_stop.rescale('s').magnitude]]
    handles = {}
    sorted_asigs = sorted(segment.analogsignals, key=lambda x: int(x.annotations['physical_channel_index']))
    for asig_count, asig in enumerate(sorted_asigs):
        handle, = ax.plot(sampling_times[idx[0]:idx[1]],
                          sc.stats.zscore(asig.magnitude[idx[0]:idx[1]]) + asig_count * scaling,
                          linewidth=1, color=asig.annotations['electrode_color'],
                          label=asig.annotations['cortical_location'])
        handles[asig.annotations['cortical_location']] = handle

    ax.set_yticks(np.arange(len(segment.analogsignals)) * scaling)
    ax.set_yticklabels([asig.annotations['physical_channel_index'] + 1 for asig in sorted_asigs])
    ax.set_ylabel('physical channel index')
    ax.set_xlabel('time [s]')
    plt.legend([handle for handle in handles.values()],
               [location for location in handles.keys()], loc=1)
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--format",    nargs=1, type=str, default='eps')
    CLI.add_argument("--t_start",   nargs=1, type=float, default=0)
    CLI.add_argument("--t_stop",    nargs=1, type=float, default=0)
    CLI.add_argument("--scaling",   nargs=1, type=float, default=12)
    CLI.add_argument("--show_figure",   nargs=1, type=int, default=0)

    args = CLI.parse_args()

    segment = load_segment(filename=args.data[0])

    plot_signal_traces(segment, t_start=args.t_start[0]*pq.s,
                       t_stop=args.t_stop[0]*pq.s,
                       scaling=args.scaling[0])

    if args.show_figure[0]:
        plt.show()

    plt.savefig(fname=args.output[0], format=args.format[0])
