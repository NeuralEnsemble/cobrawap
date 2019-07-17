import neo
import argparse
import os
import quantities as pq
import re
import numpy as np

def enrich(segment, grid_size, electrode_location, electrode_color,
           annotation_name, electrode_coordinates, file_origin):
    for (i, asig) in enumerate(segment.analogsignals):
        segment.analogsignals[i].file_origin = file_origin
        channel = asig.annotations[annotation_name]
        if isinstance(channel, (list, np.ndarray)):
            channel = channel[0]
        for element in electrode_location:
            if int(channel) in electrode_location[element]:
                segment.analogsignals[i].annotations['cortical_location'] = \
                                        element
                segment.analogsignals[i].annotations['electrode_color'] = \
                                        electrode_color[element]
                segment.analogsignals[i].annotations['coordinates'] = \
                                        electrode_coordinates[str(channel)]
                segment.analogsignals[i].annotations['grid_size'] = \
                                        grid_size
    return segment


def str2dict(str_list):
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


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--electrode_location", type=str, nargs='+')
    CLI.add_argument("--electrode_color", type=str, nargs='+')
    CLI.add_argument("--annotation_name", type=str)
    CLI.add_argument("--coordiantes", type=str, nargs='+')
    CLI.add_argument("--grid_size", type=int, nargs='+')

    args = CLI.parse_args()

    electrode_location = str2dict(args.electrode_location)
    electrode_color = str2dict(args.electrode_color)
    electrode_coords = str2dict(args.coordiantes)

    io = neo.Spike2IO(args.data)
    segment = io.read_segment()

    segment = enrich(segment,
                     grid_size=args.grid_size,
                     electrode_location=electrode_location,
                     electrode_color=electrode_color,
                     electrode_coordinates=electrode_coords,
                     annotation_name=args.annotation_name,
                     file_origin=args.data)

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    block = neo.core.Block(name='Results of {}'\
                                .format(os.path.basename(__file__)))
    block.segments.append(segment)
    with neo.NixIO(args.output) as io:
        io.write_block(block)
