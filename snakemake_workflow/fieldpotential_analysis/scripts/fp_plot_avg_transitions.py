import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_avg_transisitons(logMUA, state_vector, slope_window):

    up_transitions = []
    down_transitions = []

    MUA_time_res = (logMUA.t_stop - logMUA.t_start) / len(logMUA)

    for (i, bin_state) in enumerate(state_vector):
        if state_vector[i-1] != bin_state and i > slope_window and i < len(state_vector)-slope_window:
            if bin_state and logMUA[i-slope_window:i+slope_window].any():
                up_transitions += [logMUA[i-slope_window:i+slope_window]]
            elif logMUA[i-slope_window:i+slope_window].any():
                down_transitions += [logMUA[i-slope_window:i+slope_window]]
        else:
            pass
    avg_up_slope = np.mean(up_transitions, axis=0)
    avg_down_slope = np.mean(down_transitions, axis=0)

    fig, ax = plt.subplots(ncols=2, figsize=(15,8), sharey=True)

    ax[0].plot((np.arange(2*slope_window)-slope_window)*MUA_time_res, avg_up_slope, color='b')
    ax[0].fill_between((np.arange(2*slope_window)-slope_window)*MUA_time_res,
                    np.squeeze(avg_up_slope - np.std(up_transitions, axis=0)),
                    np.squeeze(avg_up_slope + np.std(up_transitions, axis=0)), color='0.8')
    ax[0].axvline(0, ls='--', color='k')
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('log(MUA)')
    ax[0].set_title('Up transition')

    ax[1].plot((np.arange(2*slope_window)-slope_window)*MUA_time_res, avg_down_slope, color='r')
    ax[1].fill_between((np.arange(2*slope_window)-slope_window)*MUA_time_res,
                    np.squeeze(avg_down_slope - np.std(down_transitions, axis=0)),
                    np.squeeze(avg_down_slope + np.std(down_transitions, axis=0)), color='0.8')
    ax[1].axvline(0, ls='--', color='k')
    ax[1].set_xlabel('time [s]')
    ax[1].set_title('Down transition')
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",            nargs='?', type=str)
    CLI.add_argument("--logMUA_estimate",   nargs='?', type=str)
    CLI.add_argument("--state_vector",      nargs='?', type=str)
    CLI.add_argument("--slope_window",      nargs='?', type=int, default=50)
    CLI.add_argument("--show_figure",       nargs='?', type=int, default=0)
    CLI.add_argument("--format",            nargs='?', type=str)
    CLI.add_argument("--channel",           nargs='?', type=int, default=1)

    args = CLI.parse_args()

    with neo.NixIO(args.logMUA_estimate) as io:
        logMUA_segment = io.read_block().segments[0]

    state_vectors = np.load(file=args.state_vector)

    plot_avg_transisitons(logMUA=logMUA_segment.analogsignals[args.channel-1],
                          state_vector=state_vectors[args.channel-1],
                          slope_window=args.slope_window)

    if args.show_figure:
        plt.show()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output, format=args.format)
