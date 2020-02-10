import numpy as np
import neo
import re
import itertools

def check_analogsignal_shape(asig):
    if type(asig) == list and len(asig) > 1:
        raise TypeError("More than one AnalogSignal found. Make sure that the "\
                      + "Segment has only one AnalogSignal of shape "\
                      + "(<time_steps>, <channels>)!")
    if type(asig) == list:
        asig = asig[0]
    if len(np.shape(np.squeeze(asig))) > 2:
        raise TypeError("AnalogSignal is not in shape (<time_steps>, <channels>)!")
    return True


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
