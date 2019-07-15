import neo
import numpy as np
import skimage as sk
import quantities as pq
import argparse
import os
import re
# re.compile('(?<=_)([0-9]+)')


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_files", nargs='+', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--sampling_rate", nargs='?', type=float)
    CLI.add_argument("--t_start", nargs='?', type=float)
    CLI.add_argument("--pixel_size", nargs='?', type=float)
    CLI.add_argument("--frame_num_regex", nargs='?', type=str)

    args = CLI.parse_args()

    # Sort filenames with ascending frame numbers
    def sort_key(file_path):
        fname = os.path.splitext(os.path.basename(file_path))[0]
        return int(re.search(args.frame_num_regex[1:-1], fname).group(0))

    sorted_filenames = sorted(args.image_files, key=sort_key)

    # Load images
    img_array = sk.img_as_float(sk.io.imread_collection(sorted_filenames,
                                                        plugin='tifffile'))

    # Store as 3D array in AnalogSignal object
    asig = neo.AnalogSignal(img_array,
                            units='dimensionless',
                            sampling_rate=args.sampling_rate*pq.Hz,
                            t_start=args.t_start*pq.s,
                            file_origin=os.path.dirname(args.image_files[0]),
                            pixel_size=args.pixel_size*pq.mm)

    # Save as NIX file
    image_block = neo.Block(name='Results of {}'\
                                 .format(os.path.basename(__file__)))
    seg = neo.Segment(name='Segment 1',
                      description='Unchanged images from {}'.format(os.path.dirname(args.image_files[0])))
    image_block.segments.append(seg)
    image_block.segments[0].analogsignals.append(asig)
    with neo.NixIO(args.output) as io:
        io.write(image_block)
