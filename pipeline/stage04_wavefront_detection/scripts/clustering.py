import neo
import numpy as np
import quantities as pq
import argparse
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--metric",      nargs='?', type=str)
    CLI.add_argument("--time_dim",   nargs='?', type=float)
    CLI.add_argument("--neighbour_distance",   nargs='?', type=float)
    CLI.add_argument("--min_samples",   nargs='?', type=int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    asig = block.segments[0].analogsignals[0]
    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if evts:
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    up_idx = np.where(evts.labels == 'UP'.encode('UTF-8'))[0]

    # build 3D array of trigger times
    triggers = np.zeros((len(up_idx), 3))
    triggers[:,2] = evts.times[up_idx] * args.time_dim
    for i, channel in enumerate(evts.array_annotations['channels'][up_idx]):
        triggers[i][0] = asig.array_annotations['x_coords'][int(channel)]
        triggers[i][1] = asig.array_annotations['y_coords'][int(channel)]

    clustering = DBSCAN(eps=args.neighbour_distance,
                        min_samples=args.min_samples,
                        metric=args.metric)
    clustering.fit(triggers)

    if len(np.unique(clustering.labels_)) < 1:
        raise ValueError("No Clusters found, please adapt the parameters!")

    # remove unclassified trigger points (label == -1)
    cluster_idx = np.where(clustering.labels_ != -1)[0]
    wave_idx = up_idx[cluster_idx]

    wave_evt = neo.Event(times=evts.times[wave_idx],
                         labels=clustering.labels_[cluster_idx],
                         name='Wavefronts',
                         array_annotations={'channels':evts.array_annotations['channels'][wave_idx],
                                            'x_coords':triggers[:,0][cluster_idx],
                                            'y_coords':triggers[:,1][cluster_idx]},
                         description='Transitions from down to up states. '\
                                    +'Labels are ids of wavefronts. '
                                    +'Annotated with the channel id ("channels") and '\
                                    +'its position ("x_coords", "y_coords").',
                         spatial_scale=asig.annotations['spatial_scale'],
                         cluster_algorithm='sklearn.cluster.DBSCAN',
                         cluster_eps=args.neighbour_distance,
                         cluster_metric=args.metric,
                         cluster_min_samples=args.min_samples)

    block.segments[0].events.append(wave_evt)

    with neo.NixIO(args.output) as io:
        io.write(block)
