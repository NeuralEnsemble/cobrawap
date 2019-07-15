import neo
import argparse
import os
import quantities as pq
import re


def enrich(segment, electrode_location, electrode_color,
           annotation_name, file_origin):
    for (i, asig) in enumerate(s
    egment.analogsignals):
        segment.analogsignals[i].file_origin = file_origin
        for element in electrode_location:
            if int(asig.annotations[annotation_name][0]) \
                    in electrode_location[element]:
                segment.analogsignals[i].annotations['cortical_location'] = element
                segment.analogsignals[i].annotations['electrode_color'] = electrode_color[element]
    else:
        pass
    return segment


def str2dict(str_list):
    my_dict = {}
    all_values = ' '.join(str_list)
    if '[' in all_values:
        for kv in all_values[1:-1].split("],"):
            k,v = kv.split(":")
            v = v.replace('[', '').replace(']', '')
            my_dict[k.strip()] = [int(val) for val in v.split(',')]
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
    args = CLI.parse_args()

    electrode_location = str2dict(args.electrode_location)
    electrode_color = str2dict(args.electrode_color)

    io = neo.Spike2IO(args.data)
    segment = io.read_segment()

    segment = enrich(segment,
                     electrode_location=electrode_location,
                     electrode_color=electrode_color,
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
