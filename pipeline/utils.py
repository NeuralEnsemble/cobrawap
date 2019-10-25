import numpy as np
import neo
import re

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

def ImageSequence2AnalogSignal(imgseq):
    # ToDo: tuple as array annotations? Or separte coords into coords_x and coords_y?
    # ToDo: map potentially 2D array annotations to 1D and update
    dim_t, dim_x, dim_y = imgseq.as_array().shape
    imgseq_flat = imgseq.as_array().reshape((dim_t, dim_x * dim_y))

    coords = np.zeros((dim_x, dim_y, 2), dtype=int)
    for x, row in enumerate(coords):
        for y, cell in enumerate(row):
            coords[x][y][0] = x
            coords[x][y][1] = y
    coords = coords.reshape((dim_x * dim_y, 2))
    coords_list = [str((c[0],c[1])) for c in coords]

    return neo.AnalogSignal(signal=imgseq_flat,
                            units=imgseq.units,
                            sampling_rate=imgseq.sampling_rate,
                            file_origin=imgseq.file_origin,
                            description=imgseq.description,
                            array_annotations={'coords': coords_list},
                            grid_size= (dim_x, dim_y),
                            **imgseq.annotations)
