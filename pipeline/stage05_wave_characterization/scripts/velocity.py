import neo
import numpy as np
import quantities as pq
import argparse
import scipy

def calc_velocity(times, locations):
    slope, _, _, _, stderr = scipy.stats.linregress(times, locations)
    return slope, stderr

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    evts = [ev for ev in block.segments[0].events if ev.name== 'Wavefronts'][0]

    velocities = np.zeros((len(np.unique(evts.labels)), 2))
    # loop over waves
    for i, wave_i in enumerate(np.unique(evts.labels)):
        idx = np.where(evts.labels == wave_i)[0]
        vx, vx_err = calc_velocity(evts.times[idx].magnitude,
                                   evts.array_annotations['x_coords'][idx])
        vy, vy_err = calc_velocity(evts.times[idx].magnitude,
                                   evts.array_annotations['y_coords'][idx])
        velocities[i] = (np.sqrt(vx**2 + vy**2),
                         np.sqrt(vx_err**2 + vy_err**2))

    np.save(args.output, velocities)
