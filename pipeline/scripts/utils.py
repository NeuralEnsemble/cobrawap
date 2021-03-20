"""
ToDo: Split up script into topics (parse_utils, io_utils, ...)?
"""
import numpy as np
import neo
import re
import itertools
import random
import os
import warnings
import quantities as pq
import matplotlib.pyplot as plt

# def check_analogsignal_shape(asig):
#     if type(asig) == list and len(asig) > 1:
#         raise TypeError("More than one AnalogSignal found. Make sure that the "\
#                       + "Segment has only one AnalogSignal of shape "\
#                       + "(<time_steps>, <channels>)!")
#     if type(asig) == list:
#         asig = asig[0]
#     if len(np.shape(np.squeeze(asig))) > 2:
#         raise TypeError("AnalogSignal is not in shape (<time_steps>, <channels>)!")
#     return True

def remove_annotations(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i, obj in enumerate(objects):
        for k in del_keys:
            if k in objects[i].annotations:
                del objects[i].annotations[k]
    return None

def flip_image(imgseq, axis=-1):
    # spatial axis 0 (~ 1) -> vertical
    # spatial axis 1 (~ 2)-> horizontal
    if len(imgseq.shape)==3 and axis==0:
        warnings.warn("Can not flip along time axis!"
                      "Interpreting axis=0 as first spatial axis (i.e. axis=1).")
        axis=1

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


def guess_type(string):
    try:
        out = int(string)
    except:
        try:
            out = float(string)
        except:
            out = str(string)
            if out == 'None':
                out = None
            elif out == 'True':
                out = True
            elif out == 'False':
                out = False
    return out


def str2dict(string):
    """
    Transforms a str(dict) back to dict
    """
    if string[0] == '{':
        string = string[1:]
    if string[-1] == '}':
        string = string[:-1]
    my_dict = {}
    # list or tuple values
    brackets = [delimiter for delimiter in ['[',']','(',')']
                if delimiter in string]
    if len(brackets):
        for kv in string.split("{},".format(brackets[1])):
            k,v = kv.split(":")
            v = v.replace(brackets[0], '').replace(brackets[1], '')
            values = [guess_type(val) for val in v.split(',')]
            if len(values) == 1:
                values = values[0]
            my_dict[k.strip()] = values
    # scalar values
    else:
        for kv in string.split(','):
            k,v = kv.split(":")
            my_dict[k.strip()] = guess_type(v.strip())
    return my_dict


def parse_string2dict(kwargs_str, **kwargs):
    if kwargs_str is None:
        return None
    my_dict = {}
    kwargs = ''.join(kwargs_str)[1:-1]
    # match all nested dicts
    pattern = re.compile("[\w\s]+:{[^}]*},*")
    for match in pattern.findall(kwargs):
        nested_dict_name, nested_dict = match.split(":{")
        nested_dict = nested_dict[:-1]
        my_dict[nested_dict_name] = str2dict(nested_dict)
        kwargs = kwargs.replace(match, '')
    # match entries with word value, list value, or tuple value
    pattern = re.compile("[\w\s]+:(?:[\w\.\s\/\-]+|\[[^\]]+\]|\([^\)]+\))")
    for match in pattern.findall(kwargs):
        my_dict.update(str2dict(match))
    return my_dict


def ordereddict_to_dict(input_dict):
    if isinstance(input_dict, dict):
        for k, v in input_dict.items():
            if isinstance(v, dict):
                input_dict[k] = ordereddict_to_dict(v)
        return dict(input_dict)
    else:
        return input_dict


def parse_plot_channels(channels, input_file):
    channels = channels if isinstance(channels, list) else [channels]
    channels = [none_or_int(channel) for channel in channels]
    # ToDo: check is channel exsits, even when there is no None
    if None in channels:
        dim_t, channel_num = load_neo(input_file, object='analogsignal',
                                      lazy=True).shape
        for i, channel in enumerate(channels):
            if channel is None or channel >= channel_num:
                channels[i] = random.randint(0,channel_num-1)
    return channels


def time_slice(neo_obj, t_start=None, t_stop=None,
               lazy=False, channel_indexes=None, unit=pq.s):
    """
    Robustly time-slices neo.AnalogSignal, neo.IrregularSampledSignal, neo.ImageSequence, or neo.Event,
    with `t_start` and `t_stop` given in seconds.
    """
    if not lazy and not hasattr(neo_obj, 'time_slice'):
        raise TypeError(f"{neo_obj} has no function 'time_slice'!")
    if t_start is None and t_stop is None:
        return neo_obj

    def robust_t(neo_obj, t_value=None, t_name='t_start', unit=unit):
        if t_value is None:
            if hasattr(neo_obj, t_name):
                t_value = getattr(neo_obj, t_name).rescale('s').magnitude
            else:
                raise Warning("t_start is not defined by the input or the object!")
        else:
            if isinstance(t_value, pq.Quantity):
                t_value = t_value.rescale('s').magnitude
            if hasattr(neo_obj, t_name):
                if not (neo_obj.t_start <= t_value <= neo_obj.t_stop):
                    t_value = getattr(neo_obj, t_name).rescale('s').magnitude
        return t_value*unit

    t_start = robust_t(neo_obj, t_start, t_name='t_start')
    t_stop = robust_t(neo_obj, t_stop, t_name='t_stop')

    if lazy and hasattr(neo_obj, 'load'):
        return neo_obj.load(time_slice=(t_start, t_stop),
                            channel_indexes=channel_indexes)
    else:
        return neo_obj.time_slice(t_start, t_stop)


def none_or_X(value, type):
    if value is None:
        return None
    try:
        return type(value)
    except ValueError:
        return None

none_or_int = lambda v: none_or_X(v, int)
none_or_float = lambda v: none_or_X(v, float)
none_or_str = lambda v: none_or_X(v, str)
str_list = lambda v: s.split(',')

def get_param(config, param_name):
    if param_name in config:
        return config[param_name]
    else:
        return None

def determine_spatial_scale(coords):
    coords = np.array(coords)
    dists = np.diff(coords[:,0])
    dists = dists[np.nonzero(dists)]
    return np.min(dists)

def determine_dims(coords):
    # spatial_scale = determine_spatial_scale(coords)
    # int_coords = np.round(np.array(coords)/spatial_scale).astype(int)
    int_coords = np.round(np.array(coords)).astype(int)
    dim_x, dim_y = np.max(int_coords[:,0])+1, np.max(int_coords[:,1])+1
    return dim_x, dim_y

def ImageSequence2AnalogSignal(block):
    # ToDo: map potentially 2D array annotations to 1D and update
    for seg_count, segment in enumerate(block.segments):
        for imgseq in segment.imagesequences:
            dim_t, dim_x, dim_y = imgseq.as_array().shape
            # coords = np.zeros((dim_x, dim_y, 2), dtype=int)
            # for x, row in enumerate(coords):
            #     for y, cell in enumerate(row):
            #         coords[x][y][0] = x
            #         coords[x][y][1] = y
            # coords = coords.reshape((dim_x * dim_y, 2))
            coords = np.array(list(itertools.product(np.arange(dim_x),
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

            asig.annotations.update(imgseq.annotations)

            remove_annotations(imgseq, del_keys=['nix_name', 'neo_name'])
            block.segments[seg_count].analogsignals.append(asig)
    return block


def AnalogSignal2ImageSequence(block):
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
                break

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
    coords = np.array([(x,y) for x,y in zip(asig.array_annotations['x_coords'],
                                            asig.array_annotations['y_coords'])],
                      dtype=int)

    asig_array = asig.as_array()
    dim_t, dim_channels = asig_array.shape
    dim_x, dim_y = determine_dims(coords)

    grid_data = np.empty((dim_t, dim_x, dim_y), dtype=asig.dtype)
    grid_data.fill(np.nan)

    for channel in range(dim_channels):
        x, y = coords[channel]
        grid_data[:, x, y] = asig_array[:, channel]

    new_asig = asig.duplicate_with_new_data(grid_data.reshape((dim_t, dim_x * dim_y)))

    # insert nans into array_annotations for empty sites
    nan_idx = np.where(np.isnan(new_asig[0]))[0]
    if not len(nan_idx):
        return asig
    nan_idx -= np.arange(len(nan_idx))

    nan_values = {'int': -1, 'float': np.nan,
                  'str': 'None', 'complex': np.nan+1j*np.nan}
    for key, values in asig.array_annotations.items():
        nan_value = nan_values[get_base_type(values)]
        new_values = np.insert(values, nan_idx, nan_value)
        if type(values) == pq.Quantity:
            new_values = new_values.magnitude * values.units
        new_asig.array_annotations[key] = new_values

    coords = np.array(list(itertools.product(np.arange(dim_x), np.arange(dim_y))))
    new_asig.array_annotate(x_coords=coords[:,0], y_coords=coords[:,1])
    return new_asig


def get_base_type(datatype):
    if hasattr(datatype, 'dtype'):
        datatype = datatype.dtype
    elif not type(datatype) == type:
        datatype = type(datatype)

    if datatype == list:
        warnings.warn("List don't have a defined type!")

    if np.issubdtype(datatype, np.integer):
        return 'int'
    elif np.issubdtype(datatype, float):
        return 'float'
    elif np.issubdtype(datatype, str):
        return 'str'
    elif np.issubdtype(datatype, complex):
        return 'complex'
    else:
        warnings.warn("Did not recognize type {dtype}!")
    return None


def load_neo(filename, object='block', lazy=False, *args, **kwargs):
    try:
        io = neo.io.get_io(str(filename), 'ro', *args, **kwargs)
        if lazy and io.support_lazy:
            block = io.read_block(lazy=lazy)
        # elif lazy and isinstance(io, neo.io.nixio.NixIO):
        #     with neo.NixIOFr(filename, *args, **kwargs) as nio:
        #         block = nio.read_block(lazy=lazy)
        else:
            block = io.read_block()
    except Exception as e:
        # io.close()
        raise e
    finally:
        if not lazy and hasattr(io, 'close'):
            io.close()

    if block is None:
        raise IOError(f'{filename} does not exist!')

    if object == 'block':
        return block
    elif object == 'analogsignal':
        return block.segments[0].analogsignals[0]
    else:
        raise IOError(f"{object} not recognized! Choose 'block' or 'analogsignal'.")


def write_neo(filename, block, *args, **kwargs):
    # muting saving imagesequences for now, since they do not yet
    # support array_annotations
    block.segments[0].imagesequences = []
    try:
        io = neo.io.get_io(str(filename), *args, **kwargs)
        io.write(block)
    except Exception as e:
        print(e)
    finally:
        io.close()
    return True


def save_plot(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(fname=filename, bbox_inches='tight')
    plt.close()
    return None
