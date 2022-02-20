import neo
import numpy as np
import quantities as pq
import argparse
from sklearn.cluster import DBSCAN
from utils.io import load_neo, write_neo
from utils.neo import remove_annotations


def cluster_triggers(event, metric, neighbour_distance, min_samples,
                     time_space_ratio, sampling_rate):
    up_idx = np.where(event.labels == 'UP')[0]
    # build 3D array of trigger times
    triggers = np.zeros((len(up_idx), 3))
    triggers[:,0] = event.array_annotations['x_coords'][up_idx]
    triggers[:,1] = event.array_annotations['y_coords'][up_idx]
    triggers[:,2] = event.times[up_idx].rescale('s') \
                    * sampling_rate.rescale('Hz') * args.time_space_ratio

    clustering = DBSCAN(eps=args.neighbour_distance,
                        min_samples=args.min_samples,
                        metric=args.metric)
    clustering.fit(triggers)

    if len(np.unique(clustering.labels_)) < 1:
        raise ValueError("No clusters found, please adapt the parameters!")

    # remove unclassified trigger points (label == -1)
    cluster_idx = np.where(clustering.labels_ != -2)[0]
    if not len(cluster_idx):
        raise ValueError("Clusters couldn't be classified, please adapt the parameters!")

    wave_idx = up_idx[cluster_idx]

    evt = neo.Event(times=event.times[wave_idx],
                    labels=clustering.labels_[cluster_idx].astype(str),
                    name='wavefronts',
                    array_annotations={'channels':event.array_annotations['channels'][wave_idx],
                                       'x_coords':triggers[:,0][cluster_idx].astype(int),
                                       'y_coords':triggers[:,1][cluster_idx].astype(int)},
                    description='transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    cluster_algorithm='sklearn.cluster.DBSCAN',
                    cluster_eps=args.neighbour_distance,
                    cluster_metric=args.metric,
                    cluster_min_samples=args.min_samples)

    remove_annotations(event, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(event.annotations)
    return evt

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--metric", nargs='?', type=str, default='euclidean',
                     help="parameter for sklearn.cluster.DBSCAN")
    CLI.add_argument("--time_space_ratio", nargs='?', type=float, default=1,
                     help="factor to apply to time values")
    CLI.add_argument("--neighbour_distance", nargs='?', type=float, default=30,
                     help="eps parameter in sklearn.cluster.DBSCAN")
    CLI.add_argument("--min_samples", nargs='?', type=int, default=10,
                     help="minimum number of trigger times to form a wavefront")
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    evts = block.filter(name='transitions', objects="Event")[0]

    wave_evt = cluster_triggers(event=evts,
                                metric=args.metric,
                                neighbour_distance=args.neighbour_distance,
                                min_samples=args.min_samples,
                                time_space_ratio=args.time_space_ratio,
                                sampling_rate=asig.sampling_rate)

    block.segments[0].events.append(wave_evt)

    write_neo(args.output, block)
