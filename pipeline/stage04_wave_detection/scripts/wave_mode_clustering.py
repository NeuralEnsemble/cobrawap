import argparse
import itertools
import neo
import numpy as np
import pandas as pd
from copy import copy
from warnings import warn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from skimage.transform import resize
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.io import load_neo, write_neo, save_plot
from utils.parse import none_or_str, none_or_int
from utils.neo import analogsignals_to_imagesequences, remove_annotations


def build_timelag_dataframe(waves_evt, normalize=True):
    wave_ids = np.unique(waves_evt.labels).astype(int)
    channel_ids = np.unique(waves_evt.array_annotations['channels'])

    # init dataframe
    timelag_matrix = np.empty((len(wave_ids), len(channel_ids))) * np.nan
    timelag_df = pd.DataFrame(timelag_matrix,
                              index=wave_ids,
                              columns=channel_ids)
    timelag_df.index.name = 'wave_ids'
    timelag_df.columns.name = 'channel_ids'

    # fill timelag dataframe
    for i, trigger in enumerate(waves_evt):
        wave_id = int(trigger.labels)
        channel_id = waves_evt.array_annotations['channels'][i]
        # use only first trigger per channel and wave
        if np.isnan(timelag_df[channel_id][wave_id]):
            timelag_df[channel_id][wave_id] = trigger.magnitude

    if normalize:
        norm_func = lambda row: row - np.nanmean(row)
        timelag_df = timelag_df.apply(norm_func, axis=1)

    return timelag_df

def fill_nan_sites_from_similar_waves(timelag_df, num_neighbours=5,
                                      outlier_quantile=0.95):
    ## init arrays
    num_waves = timelag_df.index.size
    pair_indices = np.triu_indices(num_waves, 1)
    wavepair_distances = np.empty(len(pair_indices[0]), dtype=float) * np.nan
    neighbourhood_distance = np.empty(num_waves, dtype=float) * np.nan

    stds = np.array([])
    ## calculate wave distances
    for i, (a,b) in enumerate(zip(pair_indices[0], pair_indices[1])):
        wave_a, wave_b = timelag_df.iloc[a], timelag_df.iloc[b]
        wavepair_distances[i] = np.nanmean(np.abs(wave_a - wave_b))

    for row, wave_id in enumerate(timelag_df.index):
        ## sort other waves by their distance
        pair_pos = get_triu_indices_pos(i=row, N=num_waves)
        neighbour_distances = wavepair_distances[pair_pos]
        sort_idx = np.argsort(neighbour_distances)
        neighbour_distances = neighbour_distances[sort_idx]
        neighbourhood_distance[row] = np.mean(neighbour_distances[:num_neighbours])
        neighbour_pos = np.array([pair_indices[0][i] if pair_indices[0][i] != row
                                  else pair_indices[1][i] for i in pair_pos[sort_idx]])

        wave = timelag_df.loc[wave_id]
        if ~np.isnan(wave).any():
            continue
        ## average trigger times of channels of similar waves
        for channel in np.where(np.isnan(wave))[0]:
            similar_trigger = timelag_df.iloc[neighbour_pos, channel]
            similar_trigger = similar_trigger[~np.isnan(similar_trigger)]
            trigger_estimate = np.mean(similar_trigger[:num_neighbours])
            stds = np.append(stds, np.std(similar_trigger[:num_neighbours]))
            timelag_df.iloc[row, channel] = trigger_estimate

    # remove outlier waves in the quantile of wave distances
    q = np.quantile(neighbourhood_distance, outlier_quantile)
    keep_rows = np.where(neighbourhood_distance <= q)[0]
    timelag_df = timelag_df.iloc[keep_rows, :]
    return timelag_df

def get_triu_indices_pos(i, N):
    idx0_start = lambda x: x*N - np.sum(range(x+1))
    idx0 = np.array([k for k in range(idx0_start(i), idx0_start(i+1))], dtype=int)
    idx1 = np.array([i-1 + np.sum([(N-2-j) for j in range(k)])
                     for k in range(i)], dtype=int)
    return np.sort(np.append(idx0, idx1))

def pca_transform(timelag_matrix, dims=None):
    if dims is not None and dims > len(timelag_df):
        warn(f'Too few waves ({len(timelag_df)}) to peform a pca reduction '
           + f'to {dims} dims. Skipping.')
        dims = None
    # n_samples x n_features
    if type(timelag_matrix) == pd.DataFrame:
        timelag_matrix = timelag_matrix.to_numpy()
    x_scaled = StandardScaler().fit_transform(timelag_matrix)
    pca_out = PCA(n_components=dims).fit(x_scaled)
    return pca_out.transform(x_scaled)

def kmeans_cluster_waves(timelag_matrix, n_cluster=7):
    kmeans = KMeans(init="k-means++",
                    n_clusters=n_cluster,
                    tol=1e-10,
                    random_state=42,
                    algorithm='full')

    return kmeans.fit(timelag_matrix)

def build_cluster_timelag_dataframe(timelag_df, cluster_labels):
    cluster_timelag_df = pd.DataFrame(index=np.unique(cluster_labels),
                                      columns=timelag_df.columns)
    cluster_timelag_df.index.name = 'cluster_ids'
    cluster_timelag_df.columns.name = 'channel_ids'

    for cluster in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster)[0]
        avg_timelags = timelag_df.iloc[idx].mean(axis='index')
        cluster_timelag_df.loc[cluster] = avg_timelags
    return cluster_timelag_df

def arange_on_grid(df, channels, x_coords, y_coords):
    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1
    grid = np.empty((df.index.size, dim_x, dim_y)) * np.nan
    for row, wave in enumerate(df.iterrows()):
        wave = wave[1]
        for channel_id, timelag in zip(wave.index, wave):
            i = np.where(channel_id == channels)[0]
            x, y = x_coords[i], y_coords[i]
            grid[row,x,y] = timelag
    return grid

def wave_to_grid(wave_evt):
    timelag_df = build_timelag_dataframe(wave_evt)
    channels = np.unique(wave_evt.array_annotations['channels']).astype(int)
    annotation_idx = [np.argmax(wave_evt.array_annotations['channels']==channel) for channel in channels]
    x_coords = wave_evt.array_annotations['x_coords'][annotation_idx].astype(int)
    y_coords = wave_evt.array_annotations['y_coords'][annotation_idx].astype(int)
    grids = arange_on_grid(timelag_df, channels, x_coords, y_coords)
    return grids

def sample_wave_pattern(pattern_func, dim_x, dim_y, step):
    nx = round((dim_x-1)/step+1)
    ny = round((dim_y-1)/step+1)
    fx, fy = np.meshgrid(np.arange(0, nx*step, step),
                         np.arange(0, ny*step, step),
                         indexing='ij')
    fcoords = np.stack((fx,fy), axis=-1)
    fdim_x, fdim_y, _ = fcoords.shape
    wave_pattern = pattern_func(fcoords.reshape(-1,2))
    return fx, fy, wave_pattern.reshape(fdim_x, fdim_y)

def interpolate_grid(grid, smoothing=0):
    x, y = np.where(np.isfinite(grid))
    rbf_func = RBFInterpolator(np.stack((x,y), axis=-1),
                               grid[x,y],
                               neighbors=None, smoothing=smoothing,
                               kernel='thin_plate_spline', epsilon=None,
                               degree=None)
    return rbf_func

def calc_cluster_distortions(feature_matrix, cluster_indices, cluster_centers):
    cluster_labels = np.unique(cluster_indices)
    cluster_dists = np.zeros(len(cluster_labels), dtype=float)

    for i, cluster_id in enumerate(cluster_labels):
        cluster_points = feature_matrix[np.where(cluster_indices==i)[0]]
        dists = cdist(cluster_points,
                      cluster_centers[i][np.newaxis,:],
                      metric='euclidean')
        cluster_dists[i] = np.sqrt(np.mean(dists**2))
    return cluster_dists

def plot_wave_modes(wavefronts_evt, wavemodes_evt):
    cmap = mpl.cm.get_cmap('coolwarm').copy()
    cmap.set_bad(color='white')
    n_modes = wavemodes_evt.annotations['n_modes']
    fig, axes = plt.subplots(ncols=n_modes+1,
                             figsize=(n_modes*4, 5),
                             gridspec_kw={'width_ratios':[1]*n_modes+[0.1]})

    int_step_size = wavemodes_evt.annotations['interpolation_step_size']

    for i, mode_id in enumerate(wavemodes_evt.annotations['mode_labels']):
        mode_count = wavemodes_evt.annotations['mode_counts'][i]
        mode_dist = wavemodes_evt.annotations['mode_distortions'][i]
        ax = axes[i]

        waves = wavefronts_evt[wavefronts_evt.array_annotations['wavemode'] == mode_id]
        waves_grid = wave_to_grid(waves)
        wavemode = wavemodes_evt[wavemodes_evt.labels.astype(int) == mode_id]
        mode_grid = wave_to_grid(wavemode)[0]

        vminmax = np.nanmax(mode_grid)
        img = ax.imshow(np.nanmean(waves_grid, axis=0), origin='lower',
                        interpolation='nearest',
                        cmap=cmap, alpha=0.5,
                        vmin=-vminmax, vmax=vminmax)

        x, y = np.where(mode_grid)
        fx = x.reshape(mode_grid.shape) * int_step_size
        fy = y.reshape(mode_grid.shape) * int_step_size
        ctr = ax.contour(fy, fx, mode_grid, levels=9,
                         cmap=cmap, linewidths=2, alpha=1,
                         vmin=-vminmax, vmax=vminmax)
        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='both', which='both',
                       labelbottom=False, bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xlabel(f'mode #{mode_id} | {mode_count} waves | var={mode_dist:.2f}')

    cbar = plt.colorbar(img, cax=axes[-1], ticks=[-vminmax/1.5, vminmax/1.5])
    cbar.ax.set_yticklabels(['wave start', 'wave end'],
                            rotation=90, va='center')
    return None

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--min_trigger_fraction", "--MIN_TRIGGER_FRACTION",
                     nargs='?', type=float, default=.5,
                     help="minimum fraction of channels to be involved in a wave")
    CLI.add_argument("--num_wave_neighbours", "--NUM_WAVE_NEIGHBOURS",
                     nargs='?', type=int, default=5,
                     help="number of similar waves to extrapolate nans from")
    CLI.add_argument("--wave_outlier_quantile", "--WAVE_OUTLIER_QUANTILE",
                     nargs='?', type=float, default=.95,
                     help="percentage of similar waves to keep")
    CLI.add_argument("--pca_dims", "--PCA_DIMS",
                     nargs='?', type=none_or_int, default=None,
                     help="reduce wave patterns to n dimensions before kmeans clustering")
    CLI.add_argument("--num_kmeans_cluster", "--NUM_KMEANS_CLUSTER",
                     nargs='?', type=int, default=5,
                     help="number of wave modes to cluster with kmeans")
    CLI.add_argument("--interpolation_step_size", "--INTERPOLATION_STEP_SIZE",
                     nargs='?', type=float, default=.2,
                     help="grid spacing for interpolation [0,1]")
    CLI.add_argument("--interpolation_smoothing", "--INTERPOLATION_SMOOTHING",
                     nargs='?', type=float, default=0,
                     help="0: no smoothing, >0: more smoothing")
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    dim_t, num_channels = asig.shape

    waves = block.filter(name='wavefronts', objects="Event")[0]
    waves = waves[waves.labels.astype(str) != '-1']

    timelag_df = build_timelag_dataframe(waves)

    # remove nan channels
    timelag_df.dropna(axis='columns', how='all', inplace=True)

    # remove small waves
    min_trigger_num = int(args.min_trigger_fraction * timelag_df.columns.size)
    timelag_df.dropna(axis='rows', thresh=min_trigger_num, inplace=True)

    # fill in nan sites with timelags from similar waves
    timelag_df = fill_nan_sites_from_similar_waves(timelag_df,
                                num_neighbours=args.num_wave_neighbours,
                                outlier_quantile=args.wave_outlier_quantile)

    # PCA transform the timelag_matrix
    timelag_matrix_transformed = pca_transform(timelag_df, dims=args.pca_dims)

    # kmeans cluster the transformed timelag_matrix into modes
    kout = kmeans_cluster_waves(timelag_matrix_transformed,
                                n_cluster=args.num_kmeans_cluster)
    mode_ids = kout.labels_
    if len(mode_ids) != len(timelag_df):
        raise IndexError('Some waves are not assigned to a kmeans cluster!'
                      + f' {len(mode_ids)} != {len(timelag_df)}')

    mode_labels, mode_counts = np.unique(mode_ids, return_counts=True)

    mode_dists = calc_cluster_distortions(timelag_matrix_transformed,
                                          cluster_indices=mode_ids,
                                          cluster_centers=kout.cluster_centers_)

    # calculate the average timelags per mode
    mode_timelag_df = build_cluster_timelag_dataframe(timelag_df, mode_ids)
    # mode_timelag_df = pd.DataFrame(kout.cluster_centers_,
    #                                index=mode_labels,
    #                                columns=timelag_df.columns)

    # rearrange average mode timelags onto channel grid
    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    channels = np.arange(len(x_coords))
    mode_grids = arange_on_grid(mode_timelag_df, channels, x_coords, y_coords)
    n_modes, dim_x, dim_y = mode_grids.shape

    # interpolate average mode timelags as pattern on grid
    for i, cluster_grid in enumerate(mode_grids):
        pattern_func = interpolate_grid(cluster_grid, args.interpolation_smoothing)
        fx, fy, pattern = sample_wave_pattern(pattern_func,
                                              step=args.interpolation_step_size,
                                              dim_x=dim_x, dim_y=dim_y)
        interpolated_mode_grids = np.concatenate((interpolated_mode_grids,
                                                     pattern[np.newaxis,:])) \
                                           if i else pattern[np.newaxis,:]

    # add cluster labels as annotation to the wavefronts event
    evt_id, waves = [(i, evt) for i, evt in enumerate(block.segments[0].events) \
                                         if evt.name=='wavefronts'][0]

    mode_annotations = np.ones(waves.size, dtype=int) * (-1)
    for wave_id, mode_id in zip(timelag_df.index, mode_ids):
        index = np.where(waves.labels.astype(int) == wave_id)[0]
        mode_annotations[index] = mode_id

    waves.array_annotate(wavemode=mode_annotations)
    block.segments[0].events[evt_id] = waves

    # add clustered wave modes as additional event 'wavemodes'
    n_modes, inter_dim_x, inter_dim_y = interpolated_mode_grids.shape
    imgseq = analogsignals_to_imagesequences(block).segments[0].imagesequences[0]
    site_grid = np.isfinite(imgseq[0].as_array())
    interpolated_site_grid = resize(site_grid,
                                    output_shape=(inter_dim_x, inter_dim_y),
                                    mode='constant', cval=True,
                                    order=0)
    ixs, iys = np.where(interpolated_site_grid)
    n_sites = len(ixs)
    mode_trigger = np.empty(n_modes*n_sites)*np.nan

    for mode_id in range(n_modes):
        for site_id, (ix, iy) in enumerate(zip(ixs, iys)):
            mode_trigger[mode_id*n_sites + site_id] \
                                    = interpolated_mode_grids[mode_id, ix, iy]

    # modes, xs, ys = np.where(np.isfinite(mode_grids))
    # channels = [np.where((asig.array_annotations['x_coords'] == x) \
    #                    & (asig.array_annotations['y_coords'] == y))[0][0] \
    #             for x,y in zip(xs,ys)]

    # remove_annotations(block.segments[0].events[evt_id])
    remove_annotations(waves)
    evt = neo.Event(mode_trigger * waves.units,
                    labels=np.repeat(mode_labels, n_sites).astype(str),
                    name='wavemodes',
                    mode_labels=mode_labels,
                    mode_counts=mode_counts[mode_labels],
                    mode_distortions=mode_dists,
                    interpolation_step_size=args.interpolation_step_size,
                    n_modes=n_modes,
                    pca_dims='None' if args.pca_dims is None else args.pca_dims,
                    **waves.annotations)
    evt.annotations['spatial_scale'] *= args.interpolation_step_size
    evt.array_annotations['x_coords'] = np.tile(ixs, n_modes).astype(int)
    evt.array_annotations['y_coords'] = np.tile(iys, n_modes).astype(int)
    evt.array_annotations['channels'] = np.tile(np.arange(n_sites), n_modes)

    block.segments[0].events.append(evt)
    # save output neo object

    write_neo(args.output, block)

    # plot and save output image
    plot_wave_modes(wavefronts_evt=block.segments[0].events[evt_id],
                    wavemodes_evt=evt)
    if args.output_img is not None:
        save_plot(args.output_img)
