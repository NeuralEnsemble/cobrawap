import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import argparse
import os
import quantities as pq


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
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--format",    nargs='?', type=str, default='eps')
    CLI.add_argument("--t_start",   nargs='?', type=float, default=0)
    CLI.add_argument("--t_stop",    nargs='?', type=float, default=0)
    CLI.add_argument("--scaling",   nargs='?', type=float, default=12)
    CLI.add_argument("--show_figure",   nargs='?', type=int, default=0)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        segment = io.read_block().segments[0]

    plot_signal_traces(segment, t_start=args.t_start*pq.s,
                       t_stop=args.t_stop*pq.s,
                       scaling=args.scaling)

    if args.show_figure[0]:
        plt.show()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output, format=args.format)
