import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import shapely.geometry as geo
import argparse
import os
import neo

def calculate_contour(img, contour_limit):
    # Computing the contour lines...
    contour_fake = sk.measure.find_contours(img, contour_limit)
    # Select the largest contour lines
    max_index = np.argmax([len(c) for c in contour_fake])

    contour = contour_fake[max_index]

    def border_intercepts(contour):
        intercept_left = False
        intercept_right = False
        intercept_top = False
        intercept_bot = False

        if not np.all(contour[:,0]):
            intercept_left = True
        if not np.all(contour[:,1]):
            intercept_top = True
        if not np.all(contour[:,0]-len(img[0])+1):
            intercept_right = True
        if not np.all(contour[:,1]-len(img[1])+1):
            intercept_bot = True
        return {'left': intercept_left, 'right': intercept_right,
                'top': intercept_top, 'bottom' : intercept_bot}

    contour_intercepts = border_intercepts(contour)
    if sum(contour_intercepts.values()) > 1:
        includes_corner = True
        del contour_fake[max_index]

        # include secondary connecting contour
        connected = False
        for snd_contour in contour_fake:
            if border_intercepts(snd_contour) == contour_intercepts:
                contour = np.append(contour, snd_contour, axis=0)
                connected = True
        # or include corner
        if not connected:
            snd_contour = []
            if contour_intercepts['left'] and contour_intercepts['top']:
                snd_contour += [0,0]
            elif contour_intercepts['top'] and contour_intercepts['right']:
                snd_contour += [len(img[0])-1,0]
            elif contour_intercepts['left'] and contour_intercepts['bottom']:
                snd_contour += [0,len(img[1]-1)]
            elif contour_intercepts['bottom'] and contour_intercepts['right']:
                snd_contour += [len(img[0])-1,len(img[1])-1]
            else:
                raise InputError('The contour is to large, and can not be determined unambigously!')
    else:
        includes_corner = False

    print('includes corner = ', includes_corner)

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

def close_contour(contour, num):
    dx = contour[-1][0] - contour[0][0]
    dy = contour[-1][1] - contour[0][1]
    avg_d = np.mean(np.append(np.abs(np.diff(contour[:,0])),
                              np.abs(np.diff(contour[:,1]))))
    num = int(np.sqrt(dx**2 + dy**2) / avg_d)
    def connect_coords(idx, num):
        connect = [contour[-1][idx], contour[0][idx]]
        return np.linspace(connect[0], connect[1], num)
    contour_connect = np.array([connect_coords(i, num)
                                for i in range(2)]).T
    return np.concatenate((contour, contour_connect))


def contour2mask(contour, dim_x, dim_y):
    mask = np.zeros((dim_x, dim_y), dtype=bool)
    polygon = geo.polygon.Polygon(contour)
    for x in range(dim_x):
        for y in range(dim_y):
            point = geo.Point(x,y)
            if polygon.contains(point):
                mask[x,y] = 1
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
    CLI.add_argument("--contour_frame", nargs='?', type=int)
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
    with neo.NixIO(args.image_file) as io:
        img_block = io.read_block()
        img_frame = img_block.segments[0].analogsignals[0][args.contour_frame]
        del img_block

    # Calculate image contour
    contour = find_contour(image=img_frame,
                           contour_limit=args.contour_limit)
    print(indent, "Mask has been found!")

    contour = close_contour(contour, num=100)

    mask = contour2mask(contour=contour,
                        dim_x=img_frame.shape[0],
                        dim_y=img_frame.shape[1])

    # Save contour, mask and example image
    if not os.path.exists(os.path.dirname(args.output_contour)):
        os.makedirs(os.path.dirname(args.output_contour))

    print_contour(img_frame, contour, show_plot=False,
                  save_path=args.output_image)

    np.save(args.output_contour, contour)

    np.save(args.output_mask, mask)
    print(indent, "Contour saved!")
