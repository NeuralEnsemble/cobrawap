import numpy as np


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
