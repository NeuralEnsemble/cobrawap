import numpy as np
import neo
import warnings
from itertools import product
import sys
from copy import copy
import quantities as pq
from pathlib import Path
from snakemake.logging import logger
from .parse import get_base_type, get_nan_value


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

def merge_analogsignals(asigs):
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

    asig_array = np.empty((min_length, len(asigs)), dtype=float) * np.nan

    for channel_number, asig in enumerate(asigs):
        asig_array[:, channel_number] = np.squeeze(asig.as_array()[:min_length])

    merged_asig = neo.AnalogSignal(asig_array*asigs[0].units,
                                   sampling_rate=asigs[0].sampling_rate,
                                   t_start=asigs[0].t_start)

    for asig in asigs:
        for key, value in asig.array_annotations.items():
            if len(value) > 1:
                warnings.warn('Unexpected length of array annotations!')
                continue
            asig.annotations[key] = value[0]

    for key in asigs[0].annotations.keys():
        annotation_values = np.array([a.annotations[key] for a in asigs])
        try:
            if (annotation_values == annotation_values[0]).all():
                merged_asig.annotations[key] = annotation_values[0]
            else:
                merged_asig.array_annotations[key] = annotation_values
        except:
            warnings.warn(f'Can not merge annotation {key}!')
    return merged_asig


def flip_image(imgseq, axis=-1):
    # spatial axis 0 (~ 1) -> vertical
    # spatial axis 1 (~ 2) -> horizontal
    if len(imgseq.shape)==3 and axis==0:
        warnings.warn("Can not flip along time axis!"
                      "Interpreting axis=0 as first spatial axis (i.e. axis=1).")
        axis = 1

    flipped = np.flip(imgseq.as_array(), axis=axis)

    flipped_imgseq = imgseq.duplicate_with_new_data(flipped)

    if 'array_annotations' in imgseq.annotations.keys():
        flipped_imgseq.annotations['array_annotations'] = {}
        for key, value in imgseq.annotations['array_annotations'].items():
            flipped_imgseq.annotations['array_annotations'][key] \
                        = np.flip(value, axis=axis)
        flipped_imgseq.annotations['array_annotations']['y_coords'] \
                    = imgseq.annotations['array_annotations']['y_coords']

    return flipped_imgseq


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

    rotated_imgseq = imgseq.duplicate_with_new_data(rotated)
    dim_t, dim_y, dim_x = rotated_imgseq.shape

    if 'array_annotations' in imgseq.annotations.keys():
        rotated_imgseq.annotations['array_annotations'] = {}
        for key, value in imgseq.annotations['array_annotations'].items():
            rotated_imgseq.annotations['array_annotations'][key] \
                    = np.rot90(value, k=nbr_of_rot90, axes=(-2,-1))

        y_coords, x_coords = np.meshgrid(range(dim_y),range(dim_x), indexing='ij')
        rotated_imgseq.annotations['array_annotations']['y_coords'] = y_coords
        rotated_imgseq.annotations['array_annotations']['x_coords'] = x_coords

    return rotated_imgseq

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


def imagesequence_to_analogsignal(imgseq):
    dim_t, dim_y, dim_x = imgseq.as_array().shape

    y_coords, x_coords = np.meshgrid(range(dim_y),range(dim_x), indexing='ij')
    x_coords = x_coords.reshape(dim_x * dim_y)
    y_coords = y_coords.reshape(dim_x * dim_y)

    imgseq_flat = imgseq.as_array().reshape((dim_t, dim_x * dim_y))

    asig = neo.AnalogSignal(signal=imgseq_flat,
                            units=imgseq.units,
                            dtype=imgseq.dtype,
                            t_start=imgseq.t_start,
                            sampling_rate=imgseq.sampling_rate,
                            file_origin=imgseq.file_origin,
                            description=imgseq.description,
                            name=imgseq.name,
                            array_annotations={'x_coords': x_coords,
                                               'y_coords': y_coords},
                            spatial_scale=imgseq.spatial_scale)

    remove_annotations(imgseq, del_keys=['nix_name', 'neo_name'])
    annotations = copy(imgseq.annotations)

    if 'array_annotations' in annotations.keys():
        try:
            for key, value in annotations['array_annotations'].items():
                asig.array_annotations[key] = value.reshape(dim_x*dim_y)

            if not (x_coords == asig.array_annotations['x_coords']).all() \
            or not (y_coords == asig.array_annotations['y_coords']).all():
                raise IndexError("Transformation of array_annotations for the "
                               + "AnalogSignal went wrong!")
        except ValueError:
            warnings.warn("ImageSequence <-> AnalogSignal transformation "
                        + "changed the signal shape!")
        del annotations['array_annotations']

    asig.annotations.update(annotations)

    return asig


def analogsignal_to_imagesequence(asig):
    asig_array = asig.as_array()
    dim_t, dim_channels = asig_array.shape

    if 'x_coords' not in asig.array_annotations \
    or 'y_coords' not in asig.array_annotations:
        logger.error('AnalogSignal has no spatial information '
                   + 'as array_annotations "x_coords" "y_coords"!')

    x_coords = asig.array_annotations['x_coords'].astype(int)
    y_coords = asig.array_annotations['y_coords'].astype(int)

    if len(x_coords) != dim_channels or len(y_coords) != dim_channels:
        logger.error("Number of channels doesn't fit number of coordinates!")

    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1

    image_data = np.empty((dim_t, dim_y, dim_x), dtype=asig.dtype) * np.nan

    for channel, (x,y) in enumerate(zip(x_coords, y_coords)):
        image_data[:, y, x] = asig_array[:, channel]

    imgseq = neo.ImageSequence(image_data=image_data,
                               units=asig.units,
                               dtype=asig.dtype,
                               t_start=asig.t_start,
                               sampling_rate=asig.sampling_rate,
                               name=asig.name,
                               description=asig.description,
                               file_origin=asig.file_origin,
                               **asig.annotations)

    # add dict of array_annotations mapped to the 2D grid
    grid_array_annotations = {}
    for key, value in asig.array_annotations.items():
        dtype = get_base_type(value[0])
        nan_value = get_nan_value(dtype)

        grid_value = np.empty((dim_y, dim_x), dtype=value.dtype)
        grid_value.fill(nan_value)

        for channel, (x,y) in enumerate(zip(x_coords, y_coords)):
            grid_value[y, x] = value[channel]

        grid_array_annotations[key] = grid_value

    ys, xs = np.meshgrid(range(dim_y),range(dim_x), indexing='ij')
    # check if -1 in xy_coords
    if not (xs == grid_array_annotations['x_coords']).all() \
    or not (ys == grid_array_annotations['y_coords']).all():
        raise ValueError("Transformation of array_annotations for the "
                       + "imagesequence went wrong!")

    imgseq.annotate(array_annotations=grid_array_annotations)
    remove_annotations(imgseq, del_keys=['nix_name', 'neo_name'])
    return imgseq


def add_empty_sites_to_analogsignal(asig):
    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1
    yx_coords = list(zip(y_coords, x_coords))

    asig_array = asig.as_array()
    dim_t, dim_channels = asig_array.shape
    num_grid_channels = dim_x * dim_y
    num_nan_channels = num_grid_channels - dim_channels

    if num_nan_channels == 0:
        return asig

    nan_signals = np.empty((dim_t, num_nan_channels)) * np.nan
    new_asig_array = np.append(asig_array, nan_signals, axis=1)

    grid_coords = list(product(range(dim_y), range(dim_x)))
    nan_coords = list(set(grid_coords).difference(yx_coords))
    y_nan_coords = np.array(nan_coords)[:,0]
    x_nan_coords = np.array(nan_coords)[:,1]
    num_nans = len(nan_coords)

    new_asig = asig.duplicate_with_new_data(new_asig_array)

    # add nans into array_annotations for empty sites
    for key, values in asig.array_annotations.items():
        nan_value = np.array([get_nan_value(get_base_type(values))])
        new_values = np.append(values, np.repeat(nan_value, num_nans))
        if type(values) == pq.Quantity:
            new_values = new_values.magnitude * values.units
        new_asig.array_annotations[key] = new_values

    new_asig.array_annotate(x_coords=np.append(x_coords, x_nan_coords),
                            y_coords=np.append(y_coords, y_nan_coords))

    return new_asig
