import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import argparse
import neo

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--output_background", nargs='?', type=str)
    CLI.add_argument("--output_image", nargs='?', type=str)
    args = CLI.parse_args()

    print('         Evaluating background...')

    # load images
    with neo.NixIO(args.image_file) as io:
        images = io.read_block().segments[0].analogsignals[0]
    # ToDo: nan to num ?

    # calculate average image as backgound
    background = np.mean(images, axis=0)
    del images

    # save background
    np.save(args.output_background, background)

    # save background image
    fig, ax = plt.subplots()
    ax.imshow(background, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.output_image)
