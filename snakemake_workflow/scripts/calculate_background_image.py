import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_files", nargs='+', type=str)
    CLI.add_argument("--output_background", nargs='?', type=str)
    CLI.add_argument("--output_image", nargs='?', type=str)
    args = CLI.parse_args()

    print('         Evaluating background...')

    # load images
    img_float = sk.img_as_float(sk.io.imread_collection(args.image_files, plugin='tifffile'))
    # ToDo: nan to num ?

    # calculate average image as backgound
    background = np.mean(img_float, axis=0)
    del img_float

    # save background
    np.save(args.output_background, background)

    # save background image
    fig, ax = plt.subplots()
    ax.imshow(background, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.output_image)
