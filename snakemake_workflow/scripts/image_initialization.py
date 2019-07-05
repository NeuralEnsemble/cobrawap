#!/usr/bin/python3
#--------1---------2---------3---------4---------5---------6---------7--------X
#  Last Update: 18.04.2019
#  Version: 1.0 (11.10.2018)
#--------1---------2---------3---------4---------5---------6---------7--------X
# 'Image_initialization' is a program to identify the interesting part of an
# image, masking all the un-needed;
# in addition, it performs background subtraction and spacial smoothing.
# An input from the user is required - see the README file for more information.

#------------------------------------ IMPORT -----------------------------------
import os
import sys, getopt
import importlib

import numpy as np

from pylab import *
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.optimize import curve_fit

import skimage
from skimage import data, io, filters, measure
from skimage import img_as_float, img_as_uint

from joblib import Parallel, delayed

from Initialization_images import drawShape, Make_a_rectangular_crop, Find_a_mask, Apply_a_mask, Inside_outside_check


#----------------------------- DEFINITION of PARAMETERS ---------------------------
print('---------------------------------- Image initialization ------------------------------------')
print('Initializing images...')


#cycle on set_data
for index, set in enumerate(SetData.num_measures):
    print('     Set Number = ', str(set))
    SET_NUMBER = str(set)

    #---------------------------------------------------------------------------
    # Here we search for the interesting part of the image. We study the first image of the whole set
    img_path = SetData.DATA_DIR + SetData.IMG_TYPE + '/t' + SET_NUMBER + '/'+ SetData.IMG_NAME + SET_NUMBER + '_' + str(SetData.TIME_MIN) + SetData.EXTENSION

    if os.path.exists(img_path) == False:
            print('ERROR:The selected IMG_PATH does not exist')
            sys.exit(-1)

    print("         Finding contours...")

    #Find contours
    print("         We want to select the interesting part of the image (Figure 1).\n We have implemented this search using the function 'measure.find_contours' of the scikit-image package: image processing in Python, https://scikit-image.org")
    print("         In order to do so, we need you to input the Contour_Limit parameter.")
    print("         Visit http://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html for more information.")

    img_float = img_as_float(io.imread_collection(img_path, plugin='tifffile'))
    #show the image row
    plt.imshow(img_float[0])
    plt.show(block=False)
    #plt.show()
    #plt.close()

    if SetData.Contour_Limit[index] == None:
        answer = 'Y'
        #the program takes in input the Contour_Limit value
        while answer == 'Y':
            Contour_Limit = input('Contour_Limit:')
            contours = Find_a_mask(img_float, Contour_Limit)
            answer = input("         Do you want to change the value? (Y/N)")
            while (answer != 'Y' and answer != 'N'):
                answer = input('         Sorry, I have not understood your answer. Answer:')
    else:
        contours = Find_a_mask(img_float, SetData.Contour_Limit[index])

    img_float = np.asarray(img_float)
    img_float = np.reshape(img_float, (SetData.DIM_X, SetData.DIM_Y))
    img_example = Apply_a_mask(img_float, contours, SetData.DIM_X, SetData.DIM_Y)
    del contours
    del img_float
    print('         Mask has been found!')

    #-------------------------------------------------------------------------------
    print('         Loading data...')
    img_path_list = []
    for i in range(SetData.TIME_MIN, SetData.TIME_MAX + 1):
        img_path = SetData.DATA_DIR + SetData.IMG_TYPE + '/t' + SET_NUMBER + '/'+ SetData.IMG_NAME + SET_NUMBER + '_' + str(i) + SetData.EXTENSION
        filename = img_path
        #filename = os.path.join(skimage.data_dir, img_path)
        img_path_list.append(filename)

    print('         Data loaded!')
    # print(img_path_list)

    #-------------------------------------------------------------------------------
    print('         Applying mask to data...')
    def Image_loading(img_path):
        # Load all the collection of the images
        img_DOWN = np.zeros((SetData.DIM_X, SetData.DIM_Y))
        img_DOWN[:,:] = np.nan
        img_float = img_as_float(io.imread_collection(img_path, plugin='tifffile'))
        img_float = np.asarray(img_float)
        img_float = np.reshape(img_float, (SetData.DIM_X, SetData.DIM_Y))
        for x in range(0,len(img_example[0])):
            for y in range(0,len(img_example[1])):
                #print(img_example[x,y]
                if not(np.isnan(img_example[x,y])):
                    img_DOWN[x,y] = img_float[x,y]

        del img_float
        #img_DOWN = Apply_a_mask(img_float, contours, DIM_X, DIM_Y)
        #print("Mask applied!")
        return(img_DOWN)

    img_collection_DOWN = []
    Y_train = []

    img_collection_DOWN.extend(Parallel(n_jobs=SetData.nprocs)(delayed(Image_loading)(img_path) for img_path in img_path_list))
    print('         Mask applied!')

    #-------------------------------------------------------------------------------
    print('         Evaluating background...')
    img_background = np.zeros((SetData.DIM_X, SetData.DIM_Y), np.float64)

    # Evaluate the background of the images as the mean over the whole set
    for i in img_collection_DOWN:
        img_background += i

    img_background /= len(img_collection_DOWN)

    # Substract from each image the background
    for i in img_collection_DOWN:
        i -= img_background
    print("         Background subtracted!")

    #-------------------------------------------------------------------------------
    img_collection_DOWN = np.asarray(img_collection_DOWN)

    print("         Spatial smoothing...")
    # Now we reduce the noise from the images by performing a spatial smoothing
    img_collection_reduced = []
    img_collection_reduced = measure.block_reduce(img_collection_DOWN, (1, SetData.MACRO_PIXEL_DIM, SetData.MACRO_PIXEL_DIM), np.mean)
    img_background = measure.block_reduce(img_background, (SetData.MACRO_PIXEL_DIM, SetData.MACRO_PIXEL_DIM), np.mean)
    #print(img_collection_reduced)
    Images_2D = np.reshape(img_collection_reduced,
                           (np.size(img_collection_reduced,0),
                            int((np.size(img_collection_reduced,1)
                                *np.size(img_collection_reduced,2)))),
                                order='C')
    Back_2D = np.reshape(img_background,
                         (int(SetData.DIM_X/SetData.MACRO_PIXEL_DIM),
                          int(SetData.DIM_Y/SetData.MACRO_PIXEL_DIM)),
                         order='C')
    del img_collection_reduced
    del img_background
    print("         Images reduced!")

    #-------------------------------------------------------------------------------

    print("         Saving images...")
    #Save initialized images
    path = SetData.ANALYSIS_DIR + SetData.IMG_TYPE + 't' + SET_NUMBER + '/'
    if os.path.exists(path) == False:
        os.makedirs(path)
    name = path + 'initialized_images_t' + SET_NUMBER + '.txt'
    #name = path + 'initialized_images_' + SetData.IMG_TYPE + '_t' + SET_NUMBER + '.txt'
    np.savetxt(name, Images_2D, delimiter = ' ', header ='#numpy 2d array containing the sequence of reduced images', newline='\n')
    #Save background
    name = path + 'background_t' + SET_NUMBER + '.txt'
    #name = path + 'background_' + SetData.IMG_TYPE + '_t' + SET_NUMBER + '.txt'
    np.savetxt( name, Back_2D, delimiter = ' ', header ='#numpy 2d array containing image background', newline='\n')

    #return Images_2D, Back_2D

print('Image initialization completed!')

plt.show(block=False)
input("Press ENTER to exit and close all the Python windows")
#return 0

if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file",   nargs='?', type=str)
    CLI.add_argument("--output_dir",   nargs='?', type=str)

    args = CLI.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(output_dir)
