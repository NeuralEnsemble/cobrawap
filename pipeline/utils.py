"""
ToDo: Split up script into topics (parse_utils, io_utils, ...)?
"""
import numpy as np
import neo
import re
import itertools
import random
import os
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
    pattern = re.compile("[\w\s]+:(?:[\w\.\s-]+|\[[^\]]+\]|\([^\)]+\))")
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


def time_slice(neo_obj, t_start=None, t_stop=None, unit='s',
               lazy=False, channel_indexes=None):
    """
    Robustly time-slices neo.AnalogSignal, neo.IrregularSampledSignal, neo.ImageSequence, or neo.Event,
    with `t_start` and `t_stop` given in seconds.
    """
    if not lazy and not hasattr(neo_obj, 'time_slice'):
        raise TypeError(f"{neo_obj} has no function 'time_slice'!")
    if t_start is None and t_stop is None:
        return neo_obj
    if hasattr(neo_obj, 't_start'):
        t_start = neo_obj.t_start.rescale('s').magnitude if t_start is None\
                  else max([t_start, neo_obj.t_start.rescale('s').magnitude])
    if hasattr(neo_obj, 't_stop'):
        t_stop = neo_obj.t_stop.rescale('s').magnitude if t_stop is None\
                 else min([t_stop, neo_obj.t_stop.rescale('s').magnitude])
    if lazy and hasattr(neo_obj, 'load'):
        return neo_obj.load(time_slice=(t_start*pq.s, t_stop*pq.s),
                            channel_indexes=channel_indexes)
    else:
        return neo_obj.time_slice(t_start*pq.s, t_stop*pq.s)


def none_or_X(value, type):
    try:
        return type(value)
    except ValueError:
        return None

none_or_int = lambda v: none_or_X(v, int)
none_or_float = lambda v: none_or_X(v, float)
none_or_str = lambda v: none_or_X(v, str)
str_list = lambda v: s.split(',')


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
                                    sampling_rate=imgseq.sampling_rate,
                                    file_origin=imgseq.file_origin,
                                    description=imgseq.description,
                                    name=imgseq.name,
                                    array_annotations={'x_coords': coords[:,0],
                                                       'y_coords': coords[:,1]},
                                    grid_size=(dim_x, dim_y),
                                    spatial_scale=imgseq.spatial_scale,
                                    **imgseq.annotations)

            chidx = neo.ChannelIndex(name=asig.name,
                                     channel_ids=np.arange(dim_x * dim_y),
                                     index=np.arange(dim_x * dim_y),
                                     coordinates=coords*imgseq.spatial_scale)

            chidx.annotations.update(asig.array_annotations)
            # asig.channel_index = chidx
            chidx.analogsignals = [asig] + chidx.analogsignals
            # block.channel_indexes.append(chidx)
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

            image_data = np.empty((dim_t, dim_x, dim_y))
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
                                       sampling_rate=asig.sampling_rate,
                                       name=asig.name,
                                       description=asig.description,
                                       file_origin=asig.file_origin,
                                       # array_annotations=array_annotations,
                                       **asig.annotations)

            block.segments[seg_count].imagesequences.append(imgseq)
    return block


def load_neo(filename, object='block', lazy=False, *args, **kwargs):
    try:
        io = neo.io.get_io(filename, *args, **kwargs)
        if lazy and io.support_lazy:
            block = io.read_block(lazy=lazy)
        # elif lazy and isinstance(io, neo.io.nixio.NixIO):
        #     with neo.NixIOFr(filename, *args, **kwargs) as nio:
        #         block = nio.read_block(lazy=lazy)
        else:
            block = io.read_block()
    except Exception as e:
        io.close()
        raise e
    finally:
        if not lazy and hasattr(io, 'close'):
            io.close()

    if object == 'block':
        return block
    elif object == 'analogsignal':
        return block.segments[0].analogsignals[0]
    else:
        raise InputError(f"{object} not recognized! Choose 'block' or 'analogsignal'.")


def write_neo(filename, block, *args, **kwargs):
    try:
        io = neo.io.get_io(filename, *args, **kwargs)
        io.write(block)
    except Exception:
        print(Exception)
    finally:
        io.close()
    return True


def save_plot(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(fname=filename, bbox_inches='tight')
    return None
