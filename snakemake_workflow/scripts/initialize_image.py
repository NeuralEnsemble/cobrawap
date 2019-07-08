import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import shapely.geometry as geo
import argparse


def substract_background(image, background):
    # ToDo: check dimensions, format, etc
    return image - background


def apply_mask(image, mask):
    image[np.bitwise_not(mask)] = np.nan
    return image


def spatial_smoothing(image, macro_pixel_dim):
    return sk.measure.block_reduce(image,
                                   (macro_pixel_dim, macro_pixel_dim),
                                   np.mean)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--background", nargs='?', type=str)
    CLI.add_argument("--mask", nargs='?', type=str)
    CLI.add_argument("--macro_pixel_dim", nargs='?', type=int)
    CLI.add_argument("--output_array", nargs='?', type=str)
    CLI.add_argument("--output_image", nargs='?', type=str)
    args = CLI.parse_args()

    image = sk.img_as_float(sk.io.imread_collection(args.image_file, plugin='tifffile'))

    image = substract_background(np.squeeze(image), np.load(args.background))

    image = apply_mask(image, np.load(args.mask))

    image = spatial_smoothing(image, args.macro_pixel_dim)

    np.save(args.output_array, image)

    # save image
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.output_image)
