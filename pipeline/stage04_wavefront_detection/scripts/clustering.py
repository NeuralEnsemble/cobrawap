import neo
import numpy as np
import quantities as pq
import argparse
from sklearn.cluster import DBSCAN
from utils import load_neo, write_neo


def cluster_triggers(event, metric, neighbour_distance, min_samples, time_dim):
    up_idx = np.where(evts.labels == 'UP'.encode('UTF-8'))[0]

    # build 3D array of trigger times
    triggers = np.zeros((len(up_idx), 3))
    triggers[:,0] = evts.array_annotations['x_coords'][up_idx]
    triggers[:,1] = evts.array_annotations['y_coords'][up_idx]
    triggers[:,2] = evts.times[up_idx] * args.time_dim
    #
    # for i, channel in enumerate(evts.array_annotations['channels'][up_idx]):
    #     triggers[i][0] = asig.array_annotations['x_coords'][int(channel)]
    #     triggers[i][1] = asig.array_annotations['y_coords'][int(channel)]

    clustering = DBSCAN(eps=args.neighbour_distance,
                        min_samples=args.min_samples,
                        metric=args.metric)
    clustering.fit(triggers)

    if len(np.unique(clustering.labels_)) < 1:
        raise ValueError("No Clusters found, please adapt the parameters!")

    # remove unclassified trigger points (label == -1)
    cluster_idx = np.where(clustering.labels_ != -1)[0]
    wave_idx = up_idx[cluster_idx]

    return neo.Event(times=evts.times[wave_idx],
                     labels=clustering.labels_[cluster_idx],
                     name='Wavefronts',
                     array_annotations={'channels':evts.array_annotations['channels'][wave_idx],
                                        'x_coords':triggers[:,0][cluster_idx],
                                        'y_coords':triggers[:,1][cluster_idx]},
                     description='Transitions from down to up states. '\
                                +'Labels are ids of wavefronts. '
                                +'Annotated with the channel id ("channels") and '\
                                +'its position ("x_coords", "y_coords").',
                     spatial_scale=evts.annotations['spatial_scale'],
                     cluster_algorithm='sklearn.cluster.DBSCAN',
                     cluster_eps=args.neighbour_distance,
                     cluster_metric=args.metric,
                     cluster_min_samples=args.min_samples)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--metric", nargs='?', type=str, default='euclidean',
                     help="parameter for sklearn.cluster.DBSCAN")
    CLI.add_argument("--time_dim", nargs='?', type=float, default=250,
                     help="factor to apply to time values")
    CLI.add_argument("--neighbour_distance", nargs='?', type=float, default=30,
                     help="eps parameter in sklearn.cluster.DBSCAN")
    CLI.add_argument("--min_samples", nargs='?', type=int, default=10,
                     help="minimum number of trigger times to form a wavefront")
    args = CLI.parse_args()

    block = load_neo(args.data)

    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if evts:
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    wave_evt = cluster_triggers(evts,
                                metric=args.metric,
                                neighbour_distance=args.neighbour_distance,
                                min_samples=args.min_samples,
                                time_dim=args.time_dim)

    block.segments[0].events.append(wave_evt)

    write_neo(args.output, block)
