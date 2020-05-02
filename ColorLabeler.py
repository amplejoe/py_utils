#!/usr/bin/env python

# +++++++++++++++++++++++++++++++++++++++++++++++++
#                         _       _
#    __ _ _ __ ___  _ __ | | ___ (_) ___   ___
#   / _` | '_ ` _ \| '_ \| |/ _ \| |/ _ \ / _ \
#  | (_| | | | | | | |_) | |  __/| | (_) |  __/
#   \__,_|_| |_| |_| .__/|_|\____/ |\___/ \___|
#                  |_|         |__/
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

# @Author: Andreas <aleibets>
# @Date: 2019-10-30T18:47:39+01:00
# @Filename: ColorLabeler.py
# @Last modified by: aleibets
# @Last modified time: 2020-01-27T12:12:00+01:00
# @description: from https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image  # (pip install Pillow)
import imutils
from . import utils

COLORS_FILE = "colors.txt"
LABELS_FILE = "labels.txt"


class ColorLabeler:
    def __init__(self, init_file_path=None):

        # default initialization of the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        self.colors = OrderedDict({
            "peritoneum": (228, 26, 28),    # red
            "ovary": (55, 126, 184),        # blue
            "die": (77, 175, 74),           # green
            "uterus": (152, 78, 163)        # violet
        })
        if init_file_path is not None:
            self.colors = self.get_colors_from_init_files(init_file_path)

        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # loop over the self.colors dictionary
        for (i, (name, rgb)) in enumerate(self.colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def get_colors_from_init_files(self,  path):
        colors_path = utils.join_paths_str(path, COLORS_FILE)
        labels_path = utils.join_paths_str(path, LABELS_FILE)
        result_dict = {}
        with open(colors_path) as colors, open(labels_path) as labels:
            for cl, la in zip(colors, labels):
                cl = list(map(int, cl.strip().split(" ")))
                la = la.strip()
                result_dict[f"{la}"] = tuple(cl)
        return result_dict

    def get_colors(self):
        return self.colors

    def label_to_color(self, label):
        return self.colors[label]

    def color_to_label(self, color):
        for k, v in self.colors.items():
            if v == color:
                return k
        return "unknown"

    def to_bgr(self, color_value):
        return tuple(reversed(color_value))

    def find_image_objects(self, image, use_bgr=False):
        """ Extracts labels, colors, bounding boxes from an image.
            input: OpenCV image
            return: array of dicts({label, color, box})
        """

        altered_img = image.copy()
        # # CROP ANNOTS and detect COLORS

        # blur the resized image slightly, then convert it to both
        # grayscale and the L*a*b* color spaces
        blurred = cv2.GaussianBlur(altered_img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        # {box, color}
        results = []
        for c in cnts:
            # detect the shape of the contour and label the color
            label = self.label(lab, c)
            color_value = self.label_to_color(label)
            if use_bgr:
                color_value = self.to_bgr(color_value)
            # box: tuple (x,y,w,h), color: tuple (r, g, b)
            results.append({'label': label, 'color': color_value,
                            'box': cv2.boundingRect(c)})

        return results

    def get_dominant_class(self, image):
        res = self.find_image_objects(image)
        all_labels = [x['label'] for x in res]
        return utils.find_most_frequent(all_labels)

    def label(self, image, c):
        """ Label single annotation image
        """
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        # initialize the minimum distance found thus far
        minDist = (np.inf, None)

        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)

            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)

        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]

    # https://stackoverflow.com/questions/54802089/convert-an-rgb-mask-image-to-coco-json-polygon-format
    def create_sub_masks(self, mask_image, bg_color=(0, 0, 0)):
        """ Takes in a mask Image object and returns a dictionary of sub-masks,
            keyed by RGB color.
        """
        width, height = mask_image.size

        # Initialize a dictionary of sub-masks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                pixel = mask_image.getpixel((x, y))[:3]

                # If the pixel is not the same as the background color
                if pixel != bg_color:
                    # Check to see if we've created a sub-mask...
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                        # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction
                        # because the contours module doesn't handle cases
                        # where pixels bleed to the edge of the image
                        sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

        return sub_masks
