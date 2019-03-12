import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import argparse
import os
import quantities as pq
from load_and_transform_to_neo import load_segment


def plot_signal_traces(logMUA, data, state_vector, t_start, t_stop, scaling=10):
    data_idx = [int(t * data.sampling_rate.rescale('1/s').magnitude)
                for t in [t_start.rescale('s').magnitude, t_stop.rescale('s').magnitude]]
    logMUA_sampling_rate = len(logMUA) / (data.t_stop - data.t_start)
    logMUA_idx = [int(t * logMUA_sampling_rate.rescale('1/s').magnitude)
                  for t in [t_start.rescale('s').magnitude, t_stop.rescale('s').magnitude]]

    logMUA_times = np.linspace(data.t_start, data.t_stop, len(logMUA))

    # ToDo: fix logMUA neo object to have .times .sampling_rate .t_start .t_stop

    up = np.max(sc.stats.zscore(data.magnitude[data_idx[0]:data_idx[1]]))
    down = np.min(sc.stats.zscore(data.magnitude[data_idx[0]:data_idx[1]]))

    fig, ax = plt.subplots()

    ax.plot(data.times[data_idx[0]:data_idx[1]],
            sc.stats.zscore(data.magnitude[data_idx[0]:data_idx[1]]),
            color='b', label='raw', lw=1)

    ax.fill_between(logMUA_times[logMUA_idx[0]:logMUA_idx[1]],
                    [down for _ in state_vector[logMUA_idx[0]:logMUA_idx[1]]],
                    [up if state else down for state in state_vector[logMUA_idx[0]:logMUA_idx[1]]],
                    color='r', alpha=0.4, label='UP state')

    ax.plot(logMUA_times[logMUA_idx[0]:logMUA_idx[1]],
            sc.stats.zscore(logMUA[logMUA_idx[0]:logMUA_idx[1]]) + scaling,
            color='r', label='log(MUA)', lw=1)

    ax.set_xlabel('time [s]')
    ax.set_yticks([])
    plt.legend()
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--logMUA",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--UD_states", nargs=1, type=str)
    CLI.add_argument("--format",    nargs=1, type=str, default='eps')
    CLI.add_argument("--t_start",   nargs=1, type=float, default=0)
    CLI.add_argument("--t_stop",    nargs=1, type=float, default=0)
    CLI.add_argument("--channel",   nargs=1, type=int, default=0)
    CLI.add_argument("--show_figure",   nargs=1, type=int, default=0)

    args = CLI.parse_args()

    data_segment = load_segment(filename=args.data[0])
    logMUA_segment = load_segment(filename=args.logMUA[0])
    state_vector = np.load(file=args.UD_states[0])

    plot_signal_traces(logMUA=logMUA_segment.analogsignals[args.channel[0]],
                       data=data_segment.analogsignals[args.channel[0]],
                       state_vector=state_vector[args.channel[0]],
                       t_start=args.t_start[0]*pq.s,
                       t_stop=args.t_stop[0]*pq.s)

    if args.show_figure[0]:
        plt.show()

    data_dir = os.path.dirname(args.output[0])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output[0], format=args.format[0])
