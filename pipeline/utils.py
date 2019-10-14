import numpy as np
import neo

def check_analogsignal_shape(asig):
    shape = np.shape(np.squeeze(asig))
    if shape[0] > 1:
        raise TypeError("Either more than one AnalogSignal found \
                         or AnalogSignal not in shape \
                         (<time_steps>, <channels>)!")
    else:
        pass
    return None


def remove_annotations(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i in range(len(objects)):
        for k in del_keys:
            if k in objects[i].annotations:
                del objects[i].annotations[k]
    return None


def str2dict(str_list):
    """
    Enables to pass a dict as commandline argument.
    """
    my_dict = {}
    all_values = ' '.join(str_list)
    # list or tuple values
    brackets = [delimiter for delimiter in ['[',']','(',')']
                if delimiter in all_values]
    if len(brackets):
        for kv in all_values[1:-1].split("{},".format(brackets[1])):
            k,v = kv.split(":")
            v = v.replace(brackets[0], '').replace(brackets[1], '')
            my_dict[k.strip()] = [int(val) for val in v.split(',')]
    # scalar values
    else:
        for kv in all_values[1:-1].split(','):
            k,v = kv.split(":")
            my_dict[k.strip()] = v.strip()
    return my_dict


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
                            **imgseq.annotations)
