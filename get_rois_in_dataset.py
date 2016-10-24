#!/usr/bin/python

# Imports
import os
import datetime
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as cpy


# Function definitions
def remove_special_chars_in_date(date_string):
    # TODO (FrAg): This should be done with a more elegant solution but... YOLO
    tmp_str = date_string
    tmp_str = tmp_str.replace(':', '_')
    tmp_str = tmp_str.replace(' ', '_')
    tmp_str = tmp_str.replace('-', '_')
    return tmp_str


def save_image_to_path(final_img, current_filename, images_path):
    cv2.imwrite(images_path + current_filename, final_img)
    return


def print_values_to_txt(current_filename, square_points, report_handler):
    for points in square_points:
        x, y, width, height = points
        report_handler.write(current_filename + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')
    return


def post_process_rois(img_roi, largest_contour=False):
    img_contours = []
    img_squares = []

    # Get Contours for the ROIs
    for roi in img_roi:
        array_roi = np.array(roi, dtype=np.uint8)
        img_roi_cont = cv2.findContours(array_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        img_contours.append(img_roi_cont)

    # Process Contours and Get Squares over the areas
    if len(img_contours) > 0:
        # Process each set of contours independently
        for cont in img_contours:
            # Get the largest contour
            # TODO (FrAg): Merge is not working yet
            cont_areas = merge_closest_contours(cont)
            if largest_contour:
                l_cont = get_largest_contour(cont_areas)
                square_points = cv2.boundingRect(l_cont)
                img_squares.append(square_points)
            else:
                for area in cont_areas:
                    square_points = cv2.boundingRect(area)
                    img_squares.append(square_points)
    return img_squares


def get_roi_of_colors(img_hsv, colors_of_interest):
    img_roi_color = []
    for color in colors_of_interest:
        range_of_interest0, range_of_interest1 = get_hsv_color_range(color)
        img_roi_color.append(get_color_roi(img_hsv, range_of_interest0, range_of_interest1))
    return img_roi_color


def get_gradient_roi(sig_borders, img_shape):
    # INFO (FrAg): As a temporal solution the ROI will be a complete area that will create a mask
    #              during the merge of ROIs the other ROIs will be taken into account only if they
    #              fall inside this large area.

    # TODO (FrAg): Implement the ROI transformation : Final ROI based on spatial gradient

    gradient_roi = np.zeros(shape=(img_shape[0], img_shape[1]))
    cv2.drawContours(gradient_roi, sig_borders, -1, (255, 255, 255), thickness=-1)

    return gradient_roi


def get_closed_contours(points, tolerance, vertices, operation):
    main_list = []
    add = False
    for elem in points:
        apx = cv2.approxPolyDP(elem, tolerance*cv2.arcLength(elem, True), True)
        if operation == 'more':
            if len(apx) > vertices:
                add = True
        if operation == 'equal':
            if len(apx) == vertices:
                add = True
        if operation == 'between':
            if vertices[0] < len(apx) < vertices[1]:
                add = True
        if add:
            main_list.append(apx)
            add = False
    return np.array(main_list)


def get_largest_contours(img_cont, perimeter, operation, number_of_contours=None):
    img_large_cont = []
    add = False

    for elem in img_cont:
        apx = cv2.arcLength(elem, True)
        if operation == 'more':
            if apx > perimeter:
                add = True
        if operation == 'equal':
            if apx == perimeter:
                add = True
        if operation == 'between':
            if perimeter[0] < apx < perimeter[1]:
                add = True
        if add:
            if number_of_contours is None:
                img_large_cont.append(elem)
            else:
                if len(img_large_cont) < number_of_contours:
                    img_large_cont.append(elem)
                else:
                    break
            add = False
    return np.array(img_large_cont)


def get_main_contours(img_contours, hierarchy):
    # Contour hierarchy is defined as: [Next, Previous, First_Child, Parent]
    main_contours = []
    for idx, contour in enumerate(img_contours):
        if hierarchy[0][idx][3] == -1:
            main_contours.append(contour)
    return main_contours


def clean_non_significant_contours(img_contours, contour_hierarchy=None, img_shape=None):
    # INFO (FrAg): Currently this function is returning complete areas where the signs could be located
    #              this helps to discriminate some ROIs.
    # TODO (FrAg): Add improvements described in the TODOs - Working
    # Function Cfg
    vector_tolerance = 0.02
    number_of_vertices = [4, 20]
    comparison_operator = 'between'
    perimeter_size = [10, 70]
    perimeter_operator = 'between'

    # TODO (FrAg): Analyse if it is necessary to use just a number of contours instead of all of them
    # number_of_contours = 1

    # Discriminates those contours that are not compliant to the specifications
    # Remove those contours contained within larger contours
    if contour_hierarchy is not None:
        main_contours = get_main_contours(img_contours, contour_hierarchy)
    else:
        main_contours = img_contours
    # Get closed contours within the vertices constraints
    # TODO (FrAg): Circles shall also be considered
    # TODO (FrAg): Fix get_closed_contours - Currently it is making the results worse
    # img_closed = get_closed_contours(main_contours, vector_tolerance, number_of_vertices, comparison_operator)
    # TODO (FrAg): Remove bridge when missing function is restored
    img_closed = main_contours
    # TODO (FrAg): Eliminate small areas : Small areas (too small to be signs) must be eliminated
    # TODO (FrAg): Remove bridge when missing function is restored
    clean_contours = img_closed

    # TODO (FrAg): This is currently not working properly, it needs to be improved and analysed
    # Get the 'number_of_contours' contours
    # TODO (FrAg): sig_contours = get_largest_contours(clean_contours, perimeter_size, perimeter_operator,
    #                                                  number_of_contours)
    # TODO (FrAg): get_largest_contours - disabled at the moment results need to be improved first
    # sig_contours = get_largest_contours(clean_contours, perimeter_size, perimeter_operator)
    # TODO (FrAg): Remove bridge when missing function is restored
    sig_contours = clean_contours

    # DEB To Remove
    # gradient_roi_tmp = np.zeros(shape=(img_shape[0], img_shape[1]))
    # cv2.drawContours(gradient_roi_tmp, sig_contours, -1, (255, 255, 255), thickness=-1)
    # cv2.imshow('ROI Window', gradient_roi_tmp)
    # End of To Remove

    return sig_contours


def get_borders(img, filter_to_use):
    # TODO (FrAg): Add other filters and their corresponding configuration
    if filter_to_use == 'Canny':
        # Getting the Canny Thresholds using Otsu's Method
        h_threshold, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        l_threshold = 0.5 * h_threshold
        # Obtain borders
        img_bord = cv2.Canny(img, l_threshold, h_threshold)
    else:
        # Not a valid filter
        img_bord = img
    return img_bord


def get_dilated_image(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype='uint8')
    dil_img = cv2.dilate(img, kernel)
    return dil_img


def get_contours(img):
    (_, img_cont, hierarchy) = cv2.findContours(np.array(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img_cont, hierarchy


def get_image_contours(grayscale_img):
    # Function Cfg
    blur_kernel = 3
    dilation_kernel = 2
    filter_to_use = 'Canny'

    # Blur image in order to improve edge recognition
    blur_img = cv2.medianBlur(grayscale_img, blur_kernel)
    # Edge detection
    img_bord = get_borders(blur_img, filter_to_use)
    # Dilation of borders
    img_dil = get_dilated_image(img_bord, dilation_kernel)
    # Get contours
    img_contours, hierarchy = get_contours(img_dil)
    return img_contours, hierarchy


def merge_my_rois(roi_color, roi_gradient):
    # TODO (FrAg): Implement Merge: Merges similar ROIs and discard those only detected by one method
    final_roi = roi_color
    return final_roi


def merge_closest_contours(contours):
    # TODO (FrAg): Implement merge of close contours : Merge those areas that have discontinuities
    cont = contours
    return cont


def get_largest_contour(contours):
    largest_cont_perimeter = 0
    largest_contour = None
    for current_cont in contours:
        curr_cont_perimeter = cv2.arcLength(current_cont, True)
        if curr_cont_perimeter > largest_cont_perimeter:
            largest_cont_perimeter = curr_cont_perimeter
            largest_contour = current_cont
    return largest_contour


def draw_squares(img, square_points):
    modified_img = cpy(img)
    # Time to print the squares over the image
    for square in square_points:
        x, y, width, height = square
        s_origin = (x, y)
        s_opposite = (x+width, y+height)
        cv2.rectangle(modified_img, s_origin, s_opposite, (0, 0, 255), 1)
    return modified_img


def get_color_roi(img_hsv, range_of_interest0, range_of_interest1=None):
    # Dilation cfg
    number_of_iterations = 8
    # Ranges cfg
    low_s = 125
    high_s = 255
    low_v = 125
    high_v = 255
    # End of cfgs
    img_roi = np.zeros(shape=(img_hsv.shape[0], img_hsv.shape[1]))
    # Only one range, the ROI is calculated only for this range
    lower_range = np.array([range_of_interest0[0], low_s, low_v])
    upper_range = np.array([range_of_interest0[1], high_s, high_v])
    roi_mask = cv2.inRange(img_hsv, lower_range, upper_range)
    roi_mask = cv2.dilate(roi_mask, None, iterations=number_of_iterations)
    img_roi[:, :] = np.array(roi_mask)
    if range_of_interest1 is not None:
        lower_range = np.array([range_of_interest1[0], low_s, low_v])
        upper_range = np.array([range_of_interest1[1], high_s, high_v])
        roi_mask0 = cv2.inRange(img_hsv, lower_range, upper_range)
        roi_mask0 = cv2.dilate(roi_mask0, None, iterations=number_of_iterations)
        np.append(img_roi, np.array(roi_mask0), axis=1)
    # DEB Delete - shows the ROI
    # cv2.imshow('ROI', img_roi)
    # end of delete
    return img_roi


def get_hsv_color_range(color):
    range0 = None
    range1 = None
    # TODO(FrAg): The desired color could be stored in a static array and the colors should be obtained in a dynamically
    # Get the H range for the desired color
    if color == 'Red':
        range0 = [0, 15]
        range1 = [160, 180]
    if color == 'Yellow':
        range0 = [20, 40]
    if color == 'Green':
        range0 = [50, 80]
    if color == 'Blue':
        range0 = [100, 130]
    return range0, range1


def show_img(img0, img1=None):
    # Plotting the image
    # If there is only one img to plot
    if img1 is None:
        plt.imshow(img0)
        plt.title('Image to Show')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Modified Image')
    plt.show()
    return


# Main procedure
def process_color_img_main(current_filename, input_images_path, output_images_path, report_handler, colors_of_interest):
    # Open Image
    img_path = input_images_path + current_filename
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Print console output
    print 'Processing image: ' + current_filename

    # Start Image Processing

    # 1. Spatial Gradient start
    # 1.1 Transform image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 1.2 Get contours and their hierarchy
    img_contours, contour_hierarchy = get_image_contours(img_gray)
    # 1.3 Discriminate some contours
    img_sig_contours = clean_non_significant_contours(img_contours, contour_hierarchy, img_gray.shape)
    # 1.4 Get the ROI based on the extracted contours
    img_roi_gradient = get_gradient_roi(img_sig_contours, img_gray.shape)
    # DEB Saving ROI to compare
    # save_image_to_path(img_roi_gradient, 'roi_' + current_filename, output_images_path)
    # END DEB
    # 2. Color Recognition Start
    # 2.1 Transform image mode to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 2.2 Get ROI based on Color
    img_roi_color = get_roi_of_colors(img_hsv, colors_of_interest)

    # 3. Merge and discriminate ROIs
    # TODO (FrAg): Merge is not working yet
    img_roi = merge_my_rois(img_roi_color, img_roi_gradient)

    # 4. Post ROI processing: Eliminates some ROIs and generate square points
    square_points = post_process_rois(img_roi, largest_contour=False)
    # 5. Output
    # 5.1 Print the values
    if len(square_points) > 0:
        print_values_to_txt(current_filename, square_points, report_handler)
    # 5.2 Draw squares around the ROIs and Save the Images
    if len(square_points) > 0:
        final_img = draw_squares(img, square_points)
        save_image_to_path(final_img, current_filename, output_images_path)
        # DEB Show Image
        # show_img(img, final_img)
        # End DEB
    return
# End of Function def


def main():
    # Argument Getter
    arguments = argparse.ArgumentParser(description="Detect ROIs based on color and gradient")
    arguments.add_argument('-i', '--input_images_path', required=True, help="Input Image Directory")
    arguments.add_argument('-e', '--input_images_extension', required=True, help="Input Image Extension e.g. .jpg")
    arguments.add_argument('-o', '--output_path', required=True, help="Output path were the reports will be generated")

    # Getting args
    arg = arguments.parse_args()
    input_images_path = arg.input_images_path
    output_path = arg.output_path
    file_ext = arg.input_images_extension

    # Creating directory structure
    # Creating output folders
    original_date = datetime.datetime.now()
    date_str = original_date.strftime('%Y_%m_%d__%H_%M_%S')
    output_path_complete = output_path + 'out\\' + 'results' + '_' + date_str + '\\'
    output_path_complete_img = output_path_complete + 'images\\'
    output_path_complete_report = output_path_complete + 'report\\'

    if not os.path.exists(output_path_complete):
        os.makedirs(output_path_complete)
        os.makedirs(output_path_complete_img)
        os.makedirs(output_path_complete_report)
    else:
        if not os.path.exists(output_path_complete_img):
            os.makedirs(output_path_complete_img)
        if not os.path.exists(output_path_complete_report):
            os.makedirs(output_path_complete_report)
    # End of creating directory structure

    # Create report file
    report_file_path = output_path_complete_report + 'report.txt'
    # Create file handler
    report_handler = open(report_file_path, 'w')
    # Report generation
    report_handler.write('***************** Report Generated on: ' + original_date.strftime('%Y-%m-%d %H:%M:%S')
                         + ' *****************\n\n')
    # End of create report file

    # Getting file list
    file_list = []
    for img in os.listdir(input_images_path):
        if img.endswith(file_ext):
            file_list.append(img)
    # End of getting file list

    # Internal Cfg
    colors_of_interest = ['Red', 'Blue', 'Yellow']
    # End of Internal Cfg
    for current_filename in file_list:
        process_color_img_main(current_filename, input_images_path, output_path_complete_img, report_handler,
                               colors_of_interest)

if __name__ == '__main__':
    main()