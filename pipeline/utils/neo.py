import numpy as np
import neo
import warnings
from itertools import product
import sys
import quantities as pq
from pathlib import Path
utils_path = str((Path(__file__).parent / '..').resolve())
sys.path.append(utils_path)
from utils.parse import determine_dims, get_base_type


def remove_annotations(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i, obj in enumerate(objects):
        for k in del_keys:
            if k in obj.annotations:
                del objects[i].annotations[k]
            if hasattr(obj, 'array_annotations') and k in obj.array_annotations:
                del objects[i].array_annotations[k]
    return None

def merge_analogsingals(asigs):
    # ToDo: to be replaced by neo utils functions
    if len(asigs) == 1:
        return asigs[0]

    min_length = np.min([len(asig.times) for asig in asigs])
    max_length = np.max([len(asig.times) for asig in asigs])
    if min_length != max_length:
        print('Warning: the length of the analog signals differs '\
            + 'between {} and {} '.format(min_length, max_length)\
            + 'All signals will be cut to the same length and merged '\
            + 'into one AnalogSignal object.')

    if len(np.unique([asig.sampling_rate for asig in asigs])) > 1:
        raise ValueError('The AnalogSignal objects have different '\
                       + 'sampling rates!')

    asig_array = np.zeros((min_length, len(asigs)))

    for channel_number, asig in enumerate(asigs):
        asig_array[:, channel_number] = np.squeeze(asig.as_array()[:min_length])

    merged_asig = neo.AnalogSignal(asig_array*asigs[0].units,
                                   sampling_rate=asigs[0].sampling_rate,
                                   t_start=asigs[0].t_start)
    for key in asigs[0].annotations.keys():
        annotation_values = np.array([a.annotations[key] for a in asigs])
        try:
            if (annotation_values == annotation_values[0]).all():
                merged_asig.annotations[key] = annotation_values[0]
            else:
                merged_asig.array_annotations[key] = annotation_values
        except:
            print('Can not merge annotation ', key)
    return merged_asig


def flip_image(imgseq, axis=-1):
    # spatial axis 0 (~ 1) -> vertical
    # spatial axis 1 (~ 2)-> horizontal
    if len(imgseq.shape)==3 and axis==0:
        warnings.warn("Can not flip along time axis!"
                      "Interpreting axis=0 as first spatial axis (i.e. axis=1).")
        axis = 1

    flipped = np.flip(imgseq.as_array(), axis=axis)

    return imgseq.duplicate_with_new_data(flipped)


def rotate_image(imgseq, rotation=0):
    # rotating clockwise
    if np.abs(rotation) <= 2*np.pi:
        # interpret as rad
        rotation = int(np.round(rotation/np.pi * 180, decimals=0))
    else:
        # interpret as deg
        pass

    nbr_of_rot90 = np.divide(rotation, 90)

    if np.mod(nbr_of_rot90, 1):
        nbr_of_rot90 = np.round(nbr_of_rot90, decimals=0)
        warnings.warn("Images can only be rotated in steps of 90 degrees. "
                     f"Rounding {rotation} deg to {nbr_of_rot90*90} deg.")

    rotated = np.rot90(imgseq.as_array(),
                       k=nbr_of_rot90,
                       axes=(-2,-1))

    return imgseq.duplicate_with_new_data(rotated)


def robust_t(neo_obj, t_value=None, t_name='t_start', unit='s'):
    if t_value is None:
        if hasattr(neo_obj, t_name):
            t_value = getattr(neo_obj, t_name).rescale('s').magnitude
        else:
            raise Warning("t_start is not defined by the input or the object!")
    else:
        if isinstance(t_value, pq.Quantity):
            t_value = t_value.rescale('s').magnitude
        if hasattr(neo_obj, t_name):
            obj_t_start = neo_obj.t_start.rescale('s').magnitude
            obj_t_stop = neo_obj.t_stop.rescale('s').magnitude
            if not (obj_t_start <= t_value <= obj_t_stop):
                t_value = getattr(neo_obj, t_name).rescale('s').magnitude
    return t_value * pq.s.rescale(unit)


def time_slice(neo_obj, t_start=None, t_stop=None,
               lazy=False, channel_indexes=None, unit=pq.s):
    """
    Robustly time-slices neo.AnalogSignal, neo.IrregularSampledSignal,
    neo.ImageSequence, or neo.Event,
    with `t_start` and `t_stop` given in seconds.
    """
    if not lazy and not hasattr(neo_obj, 'time_slice'):
        raise TypeError(f"{neo_obj} has no function 'time_slice'!")
    if t_start is None and t_stop is None:
        return neo_obj

    t_start = robust_t(neo_obj, t_start, t_name='t_start')
    t_stop = robust_t(neo_obj, t_stop, t_name='t_stop')

    if lazy and hasattr(neo_obj, 'load'):
        return neo_obj.load(time_slice=(t_start, t_stop),
                            channel_indexes=channel_indexes)
    else:
        return neo_obj.time_slice(t_start, t_stop)


def imagesequences_to_analogsignals(block):
    # ToDo: map potentially 2D array annotations to 1D and update
    for seg_count, segment in enumerate(block.segments):
        for imgseq in segment.imagesequences:
            dim_t, dim_x, dim_y = imgseq.as_array().shape

            coords = np.array(list(product(np.arange(dim_x),
                                                     np.arange(dim_y))))

            imgseq_flat = imgseq.as_array().reshape((dim_t, dim_x * dim_y))
            asig = neo.AnalogSignal(signal=imgseq_flat,
                                    units=imgseq.units,
                                    dtype=imgseq.dtype,
                                    t_start=imgseq.t_start,
                                    sampling_rate=imgseq.sampling_rate,
                                    file_origin=imgseq.file_origin,
                                    description=imgseq.description,
                                    name=imgseq.name,
                                    array_annotations={'x_coords': coords[:,0],
                                                       'y_coords': coords[:,1]},
                                    spatial_scale=imgseq.spatial_scale,
                                    )

            # chidx = neo.ChannelIndex(name=asig.name,
            #                          channel_ids=np.arange(dim_x * dim_y),
            #                          index=np.arange(dim_x * dim_y),
            #                          coordinates=coords*imgseq.spatial_scale)

            # chidx.annotations.update(asig.array_annotations)
            # asig.channel_index = chidx
            # chidx.analogsignals = [asig] + chidx.analogsignals
            # block.channel_indexes.append(chidx)
            if 'array_annotations' in imgseq.annotations.keys():
                try:
                    asig.array_annotations.update(imgseq.annotations['array_annotations'])
                except ValueError:
                    warnings.warn("ImageSequence <-> AnalogSignal transformation " \
                                + "changed the signal shape!")
                del imgseq.annotations['array_annotations']


            remove_annotations(imgseq, del_keys=['nix_name', 'neo_name'])
            asig.annotations.update(imgseq.annotations)
            block.segments[seg_count].analogsignals.append(asig)
    return block


def analogsignals_to_imagesequences(block):
    # ToDo: map 1D array annotations to 2D and update
    for seg_count, segment in enumerate(block.segments):
        for asig_count, asig in enumerate(segment.analogsignals):
            asig_array = asig.as_array()
            dim_t, dim_channels = asig_array.shape
            # coords = asig.channel_index.coordinates
            # temporary replacement
            if 'x_coords' not in asig.array_annotations\
                or 'y_coords' not in asig.array_annotations:
                print('AnalogSignal {} in Segment {} has no spatial Information '\
                      .format(asig_count, seg_count)\
                    + ' as array_annotations "x_coords" "y_coords", skip.')
                continue

            coords = np.array([(x,y) for x,y in zip(asig.array_annotations['x_coords'],
                                                    asig.array_annotations['y_coords'])],
                              dtype=float)
            #
            # spatial_scale = asig.annotations['spatial_scale']
            # int_coords = np.round(np.array(coords)/spatial_scale).astype(int)
            # print(int_coords)

            if len(coords) != dim_channels:
                raise IndexError("Number of channels doesn't agree with "\
                               + "number of coordinates!")

            dim_x, dim_y = determine_dims(coords)

            image_data = np.empty((dim_t, dim_x, dim_y), dtype=asig.dtype)
            image_data[:] = np.nan

            for channel in range(dim_channels):
                x, y = coords[channel]
                x, y = int(x), int(y)
                image_data[:, x, y] = asig_array[:, channel]

            # spatial_scale = determine_spatial_scale(coords)*coords.units
            spatial_scale = asig.annotations['spatial_scale']

            # array_annotations = {}
            # for k, v in asig.array_annotations.items():
            #     array_annotations[k] = v.reshape((dim_x, dim_y))
            imgseq = neo.ImageSequence(image_data=image_data,
                                       units=asig.units,
                                       dtype=asig.dtype,
                                       t_start=asig.t_start,
                                       sampling_rate=asig.sampling_rate,
                                       name=asig.name,
                                       description=asig.description,
                                       file_origin=asig.file_origin,
                                       # array_annotations=array_annotations,
                                       **asig.annotations)

            imgseq.annotate(array_annotations=asig.array_annotations)

            remove_annotations(imgseq, del_keys=['nix_name', 'neo_name'])
            block.segments[seg_count].imagesequences.append(imgseq)
    return block


def add_empty_sites_to_analogsignal(asig):
    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    coords = list(zip(x_coords, y_coords))

    asig_array = asig.as_array()
    dim_t, dim_channels = asig_array.shape
    dim_x, dim_y = determine_dims(coords)
    num_grid_channels = dim_x * dim_y
    num_nan_channels = num_grid_channels - dim_channels

    nan_signals = np.empty((dim_t, num_nan_channels)) * np.nan
    new_asig_array = np.append(asig_array, nan_signals, axis=1)

    grid_coords = list(product(range(dim_x), range(dim_y)))
    nan_coords = list(set(grid_coords).difference(coords))
    x_nan_coords = np.array(nan_coords)[:,0]
    y_nan_coords = np.array(nan_coords)[:,1]
    num_nans = len(nan_coords)

    new_asig = asig.duplicate_with_new_data(new_asig_array)

    # add nans into array_annotations for empty sites
    nan_values = {'int': -1, 'float': np.nan, 'bool': False,
                  'str': 'None', 'complex': np.nan+1j*np.nan}

    for key, values in asig.array_annotations.items():
        nan_value = np.array([nan_values[get_base_type(values)]])
        new_values = np.append(values, np.repeat(nan_value, num_nans))
        if type(values) == pq.Quantity:
            new_values = new_values.magnitude * values.units
        new_asig.array_annotations[key] = new_values

    new_asig.array_annotate(x_coords=np.append(x_coords, x_nan_coords), 
                            y_coords=np.append(y_coords, y_nan_coords))

    return new_asig
