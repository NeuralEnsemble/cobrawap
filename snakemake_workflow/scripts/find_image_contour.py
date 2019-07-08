import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import shapely.geometry as geo
import argparse
import os

def calculate_contour(img, contour_limit):
    # Computing the contour lines...
    contour_fake = sk.measure.find_contours(img, contour_limit)
    # Select the largest contour lines
    max_index = np.argmax([len(c) for c in contour_fake])

    contour = contour_fake[max_index]
    appoggio_updown = False
    appoggio_sxdx = False

    if not np.all(contour):
        appoggio_updown = True
    if not np.all(contour[:,0]-len(img[0]+1)) \
    or not np.all(contour[:,1]-len(img[1]+1)):
        appoggio_sxdx = True

    print('snd = ', bool(appoggio_updown*appoggio_sxdx))
    if (appoggio_updown and appoggio_sxdx):
        if max_index:
            snd_index = 0
        else:
            snd_index = 1
        # ToDo:
        # what is this supposed to do?
        # why is snd_index potentially overwritten?
        for i in range(len(contour_fake)):
            if (len(contour_fake[i]) > len(contour_fake[snd_index])
            and i!=max_index):
                snd_index = i

        for i in range(len(contour_fake[snd_index])):
            contour = np.append(contour, contour_fake[snd_index][i], axis = 0)

    print('There are contours found with contour limit = {}'.format(contour_limit))

    return contour


def print_contour(img_float, contour, ax=None, show_plot=True, save_path=None):
        # Display the image and plot all contours found
    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(img_float, interpolation='nearest', cmap=plt.cm.gray)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        if show_plot:
            plt.show(block=False)
    else:
        plt.cla()

    ax.imshow(img_float, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.draw()

    if save_path is not None:
        plt.savefig(save_path)

    return ax


def find_contour(image, contour_limit):
    if contour_limit is None:
        ax = None
        answer = 'Y'
        #the program takes in input the Contour_Limit value
        while answer == 'Y':
            input_contour_limit = input('Contour_Limit:')
            contour = calculate_contour(image, input_contour_limit)
            ax = print_contour(image, contour, ax=ax, show_plot=True)
            answer = input("         Do you want to change the value? (Y/N)")
            while (answer != 'Y' and answer != 'N'):
                answer = input('         Sorry, I have not understood your answer. Answer:')
    else:
        contour = calculate_contour(image, contour_limit)

    return contour


def contour2mask(contour, dim_x, dim_y):
    mask = np.zeros((dim_x, dim_y), dtype=bool)
    polygon = geo.polygon.Polygon(contour)
    for x in range(dim_x):
        for y in range(dim_y):
            point = geo.Point(x,y)
            if polygon.contains(point):
                mask[x,y] = 1
    print(np.where(mask))
    return mask


if __name__ == '__main__':
    def none_or_float(value):
        if value == 'None':
            return None
        return float(value)
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--output_contour", nargs='?', type=str)
    CLI.add_argument("--output_image", nargs='?', type=str)
    CLI.add_argument("--output_mask", nargs='?', type=str)
    CLI.add_argument("--contour_limit", nargs='?', type=none_or_float)
    args = CLI.parse_args()

    indent = " "*9
    print('-'*37, ' Image initialization ', '-'*37,)
    print('Initializing images...')

    if os.path.exists(args.image_file) == False:
            print('ERROR:The selected IMG_PATH does not exist')
            sys.exit(-1)

    print(indent, "We want to select the interesting part of the image (Figure 1).\n We have implemented this search using the function 'measure.find_contours' of the scikit-image package: image processing in Python, https://scikit-image.org")
    print(indent, "In order to do so, we need you to input the Contour_Limit parameter.")
    print(indent, "Visit http://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html for more information.")

    print(indent, "Finding contours...")

    # Load image
    img_float = sk.img_as_float(sk.io.imread_collection(args.image_file, plugin='tifffile'))

    # Calculate image contour
    contour = find_contour(image=img_float[0],
                           contour_limit=args.contour_limit)
    print(indent, "Mask has been found!")

    mask = contour2mask(contour=contour,
                        dim_x=img_float[0].shape[0],
                        dim_y=img_float[0].shape[1])

    # Save contour, mask and example image
    if not os.path.exists(os.path.dirname(args.output_contour)):
        os.makedirs(os.path.dirname(args.output_contour))

    print_contour(img_float[0], contour, show_plot=False, save_path=args.output_image)

    np.save(args.output_contour, contour)

    np.save(args.output_mask, mask)

    print(indent, "Contour saved!")
