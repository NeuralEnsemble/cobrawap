import argparse
import numpy as np
import scipy as sc
import neo
import os
import matplotlib.pyplot as plt


def logMUA_distribution(logMUA, fixed_threshold, sigma_threshold, plot, bins=100):
    # signal amplitude distribution
    logMUA = logMUA[np.isfinite(logMUA)]
    hist, edges = np.histogram(logMUA, bins=bins, density=True)
    xvalues = edges[:-1] + np.diff(edges)[0] / 2.

    # First Gaussian fit -> determine peak location m0
    gaussian = lambda x, m, s: 1. / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - m) / s) ** 2)
    (m0, _), _ = sc.optimize.curve_fit(gaussian, xvalues, hist, p0=(-4, 1))

    # shifting to 0
    xvalues -= m0

    # Mirror left peak side for 2nd Gaussian fit
    logMUA_leftpeak = logMUA[logMUA - m0 <= 0] - m0
    left_right_ratio = len(logMUA_leftpeak) * 2. / len(logMUA)
    logMUA_peak = np.append(logMUA_leftpeak, -1 * logMUA_leftpeak)
    peakhist, edges = np.histogram(logMUA_peak, bins=bins, density=True)
    xvalues2 = edges[:-1] + np.diff(edges)[0] / 2.

    # Second Gaussian fit -> determine spread s0
    (_, s0), _ = sc.optimize.curve_fit(gaussian, xvalues2, peakhist, p0=(0, 1))

    ## PLOTTING ## ToDO: outsource?
    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
        ax[0].bar(xvalues, hist, width=np.diff(xvalues)[0], color='r')
        ax[0].plot(xvalues, [left_right_ratio * gaussian(x, 0, s0) for x in xvalues], c='k')
        ax[0].set_xlabel('log(MUA)')
        ax[0].set_ylabel('sample density')
        ax[0].set_title('Amplitude distribution')

        ax[1].bar(xvalues, [hist[i] - gaussian(x, 0, s0) for (i, x) in enumerate(xvalues)],
                  width=np.diff(xvalues)[0], color='r')
        ax[1].set_xlabel('log(MUA)')
        ax[1].set_title('Non-Gaussian tail')
        if fixed_threshold:
            ax[1].axvline(fixed_threshold, color='k', ls='--'),
            ax[1].text(1.1 * fixed_threshold, 0.9 * ax[1].get_ylim()[0],
                       r'UD threshold ({})'.format(fixed_threshold), color='k')
        if not fixed_threshold and sigma_threshold:
            ax[1].axvline(sigma_threshold * s0, color='k', ls='--'),
            ax[1].text(1.1 * sigma_threshold * s0, 0.9 * ax[1].get_ylim()[0],
                       r'UD threshold ({}$\sigma$)'.format(sigma_threshold), color='k')
        plt.show()
    return m0, s0


def remove_short_states(state_vector, min_state_duration):
    for (i, bin_state) in enumerate(state_vector[:-min_state_duration - 1]):
        if bin_state != state_vector[i + 1] \
                and bin_state == state_vector[i + 1 + min_state_duration]:
            state_vector[i:i + 1 + min_state_duration] = [bin_state] * (min_state_duration + 1)
        else:
            pass
    return None


def create_state_vector(logMUA, fixed_threshold, sigma_threshold, plot):
    m0, s0 = logMUA_distribution(logMUA, fixed_threshold=fixed_threshold,
                                 sigma_threshold=sigma_threshold, plot=plot)
    if fixed_threshold:
        threshold = fixed_threshold + m0
    else:
        threshold = sigma_threshold * s0 + m0

    state_vector = [True if value > threshold else False for value in logMUA]

    return state_vector


def create_all_state_vectors(logMUA_signals, min_state_duration,
                             fixed_threshold=0, sigma_threshold=0,
                             plot=False):
    state_vectors = []
    UP_transitions = []
    DOWN_transitions = []
    for i, asig in enumerate(logMUA_signals):
        state_vector = create_state_vector(asig.magnitude,
                                           fixed_threshold=fixed_threshold,
                                           sigma_threshold=sigma_threshold,
                                           plot=plot)
        remove_short_states(state_vector, min_state_duration)
        ups, downs = statevector_to_spiketrains(state_vector,
                                                t_start=asig.t_start,
                                                t_stop=asig.t_stop,
                                                sampling_rate=asig.sampling_rate,
                                                fixed_threshold=fixed_threshold,
                                                sigma_threshold=sigma_threshold,
                                                min_state_duration=min_state_duration,
                                                **asig.annotations)
        UP_transitions += [ups]
        DOWN_transitions += [downs]
        state_vectors += [state_vector]
    # ToDo: write UD states in neo segment as epochs?
    return np.array(state_vectors), UP_transitions, DOWN_transitions


def statevector_to_spiketrains(state_vector, sampling_rate, t_start, t_stop,
                               **annotations):
    ups = np.array([])
    downs = np.array([])
    for i, state in enumerate(state_vector[:-1]):
        # Transition
        if not state*state_vector[i+1]:
            # UP -> DOWN
            if state:
                downs = np.append(downs, i+0.5)
            # DOWN -> UP
            else:
                ups = np.append(ups, i+0.5)
    downs = downs/sampling_rate
    ups = ups/sampling_rate
    up_trains = neo.core.SpikeTrain(ups,
                                    t_start=t_start,
                                    t_stop=t_stop,
                                    sampling_rate=sampling_rate,
                                    **annotations)
    down_trains = neo.core.SpikeTrain(downs,
                                      t_start=t_start,
                                      t_stop=t_stop,
                                      sampling_rate=sampling_rate,
                                      **annotations)
    return up_trains, down_trains


def remove_duplicate_properties(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i in range(len(objects)):
        for k in del_keys:
            if k in objects[i].annotations:
                del objects[i].annotations[k]
    return None

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--out_state_vector",    nargs='?', type=str)
    CLI.add_argument("--out_nix_file",    nargs='?', type=str)
    CLI.add_argument("--logMUA_estimate",      nargs='?', type=str)
    CLI.add_argument("--min_state_duration",  nargs='?', type=int, default=2)
    CLI.add_argument("--fixed_threshold", nargs='?', type=int, default=0)
    CLI.add_argument("--sigma_threshold",  nargs='?', type=int, default=0)
    CLI.add_argument("--show_plots",  nargs='?', type=int, default=0)
    args = CLI.parse_args()

    with neo.NixIO(args.logMUA_estimate) as io:
        logMUA_block = io.read_block()

    remove_duplicate_properties(logMUA_block.segments[0].analogsignals)
    remove_duplicate_properties([logMUA_block, logMUA_block.segments[0]])

    logMUA_signals = logMUA_block.segments[0].analogsignals

    state_vectors, up_trains, down_trains = create_all_state_vectors(
                                            logMUA_signals,
                                            min_state_duration=args.min_state_duration,
                                            fixed_threshold=args.fixed_threshold,
                                            sigma_threshold=args.sigma_threshold,
                                            plot=args.show_plots)

    np.save(args.out_state_vector, state_vectors)

    logMUA_block.name += 'and {}'.format(os.path.basename(__file__))
    seg1 = neo.Segment(name='Segment DOWN -> UP',
                       description='Transitions from UP to DOWN state')
    seg2 = neo.Segment(name='Segment UP -> DOWN',
                       description='Transitions from DOWN to UP state')

    seg1.spiketrains = up_trains
    seg2.spiketrains = down_trains

    logMUA_block.segments.append(seg1)
    logMUA_block.segments.append(seg2)

    with neo.NixIO(args.out_nix_file) as io:
        io.write(logMUA_block)
