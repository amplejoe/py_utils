#!/usr/bin/env python

###
# File: ColorLabeler.py
# Created: Tuesday, 12th May 2020 5:46:35 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:12:38 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###


# import the necessary packages
import typing
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
from . import opencv_utils
from PIL import Image  # (pip install Pillow)

# import imutils
import copy
from yattag.doc import Doc  # for html out
from . import utils

COLORS_FILE = "colors.txt"
LABELS_FILE = "labels.txt"
HTML_FILE = "label_colors.html"
BACKGROUND_LABELS = ["background", "bg"]

# DEFAULT_COLORS = OrderedDict(
#     {
#         "background": (0, 0, 0),  # black (bg)
#         "peritoneum": (228, 26, 28),  # red
#         "ovary": (55, 126, 184),  # blue
#         "die": (77, 175, 74),  # green
#         "uterus": (152, 78, 163),  # violet
#     }
# )
DEFAULT_COLORS = OrderedDict(
    {
        "background": (0, 0, 0),  # black (bg)
        "default": (255, 255, 255),  # white (fg)
    }
)


class ColorLabeler:
    def __init__(self, init_file_path=None, init_color_dict=DEFAULT_COLORS):

        # default initialization of the colors dictionary (if no init_file_path given), containing the color
        # name as the key and the RGB tuple as the value
        self.colors = init_color_dict
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

    def get_colors_from_init_files(self, path):
        colors_path = utils.join_paths_str(path, COLORS_FILE)
        labels_path = utils.join_paths_str(path, LABELS_FILE)
        result_dict = {}
        with open(colors_path) as colors, open(labels_path) as labels:
            for cl, la in zip(colors, labels):
                cl = list(map(int, cl.strip().split(" ")))
                la = la.strip()
                result_dict[f"{la}"] = tuple(cl)
        return result_dict

    def get_colors(self, include_bg=True):
        """Returns deep copy of colors, optionally without background

        Args:
            include_bg (bool, optional): [description]. Defaults to True.

        Returns:
            OrderedDict: color dictionary
        """
        ret = copy.deepcopy(self.colors)
        if not include_bg:
            for n in BACKGROUND_LABELS:
                if n in ret:
                    del ret[n]
        return ret

    def get_labels(self, include_bg=True):
        ret = list(self.colors.keys())
        if not include_bg:
            ret = [x for x in ret if x not in BACKGROUND_LABELS]
        return ret

    def label_to_color(self, label):
        if label not in self.colors:
            if label == 'background':
                return DEFAULT_COLORS['background']
            else:
                return DEFAULT_COLORS['default']
        return self.colors[label]

    def color_to_label(self, color):
        for k, v in self.colors.items():
            if v == color:
                return k
        return "unknown"

    def exist_color_files(self, path):
        out_color = utils.join_paths_str(path, COLORS_FILE)
        out_label = utils.join_paths_str(path, LABELS_FILE)
        html_file = utils.join_paths_str(path, HTML_FILE)
        return (
            utils.exists_file(out_color)
            and utils.exists_file(out_label)
            and utils.exists_file(html_file)
        )

    def save_colors(self, out_path):
        out_color = utils.join_paths_str(out_path, COLORS_FILE)
        out_label = utils.join_paths_str(out_path, LABELS_FILE)
        html_file = utils.join_paths_str(out_path, HTML_FILE)
        utils.confirm_overwrite(out_color, "n")
        utils.confirm_overwrite(out_label, "n")
        utils.confirm_overwrite(html_file, "n")

        with open(out_color, "w", newline="") as c_out, open(
            out_label, "w", newline=""
        ) as l_out:
            for label, color in self.colors.items():
                l_out.write(f"{label}\n")
                c_out.write(" ".join(str(v) for v in color) + "\n")
        print(f"Wrote {out_label}")
        print(f"Wrote {out_color}")

        # HTML out (.html)
        doc, tag, text = Doc().tagtext()
        with tag("html"):
            with tag("body"):
                with tag("table"):
                    for label, color in self.colors.items():
                        with tag("tr"):
                            css_color = f"rgb{color}"
                            with tag("td", style=f"background-color:{css_color};"):
                                # blend mode: guarantess visibility of label but fucks up looks
                                # with tag('span', style="color:white;mix-blend-mode: difference;"):
                                # -> us shadow instead (https://jsfiddle.net/dimitriadisg/agnfz7qt/)
                                with tag(
                                    "span",
                                    style="color:white;text-shadow: 0.05em 0 black, 0 0.05em black, -0.05em 0 black, 0 -0.05em black, -0.05em -0.05em black, -0.05em 0.05em black, 0.05em -0.05em black, 0.05em 0.05em black;",
                                ):
                                    text(label)
        result = doc.getvalue()
        with open(html_file, "w") as hwriter:
            hwriter.write(result)
        print(f"Wrote {html_file}")

    def mask_from_contour(self, contour, width, height, num_type=np.uint8):
        """1 channel mask from contour(s) with custom color"""
        c_image = np.zeros((height, width, 1), num_type)
        # only draws contour outline
        # cv2.drawContours(c_image, [c], 0, (255), 1)
        cv2.fillPoly(c_image, pts=[contour], color=(255))
        return c_image

    def mask_from_contours(self, contours, width, height):
        """1 channel mask from contour(s)"""
        final_img = np.zeros((height, width, 1), np.uint8)
        for c in contours:
            final_img += self.mask_from_contour(c, width, height)
        return final_img

    def image_from_contours(self, contours, width, height, color=(255, 255, 255)):
        final_img = np.zeros((height, width, 3), np.uint8)
        for c in contours:
            final_img += self.image_from_contour(c, width, height, color)
        return final_img

    def image_from_contour(self, contour, width, height, color=(255, 255, 255)):
        """3 channel mask from contour(s) with custom color"""
        c_image = np.zeros((height, width, 3), np.uint8)
        # only draws contour outline
        # cv2.drawContours(c_image, [c], 0, (255,255,255), 3)
        cv2.fillPoly(c_image, pts=[contour], color=color)
        return c_image

    def show_contours(self, contours, width, height):
        img = self.image_from_contours(contours, width, height)
        win_title = f"contours ({len(contours)})"
        cv2.imshow(win_title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)

    # for imgaug of masks
    def to_segmap(self, annot_or_path) -> typing.Tuple[np.ndarray, OrderedDict]:

        annot = opencv_utils.get_image(annot_or_path)

        # annot needs to be converted into a map (int32, w x h),
        # with labels starting from 1 (0=bg)

        # get annots and colors
        img_objects = self.find_image_objects(annot)
        h = annot.shape[0]
        w = annot.shape[1]

        segmap = np.zeros((h, w, 1), dtype=np.int32)
        color_class_map = OrderedDict()
        # 0 is reserved for background
        color_class_map["(0,0,0)"] = 0
        cur_class = 0
        for obj in img_objects:
            color = obj["color"]
            color_string = f"{color}"
            cnt = obj["contour"]
            if color_string not in color_class_map.keys():
                cur_class = len(color_class_map.keys())
                color_class_map[color_string] = cur_class
            else:
                cur_class = color_class_map[color_string]

            cv2.fillPoly(segmap, pts=[cnt], color=(cur_class))

        return segmap, color_class_map

    # for imgaug of masks
    # def from_segmap(self, segmap, color_class_map):
    #     print(f"shape before: {segmap.shape}")
    #     segmap_3channel = np.stack((segmap,) * 3, axis=-1)
    #     print(f"shape after: {segmap.segmap_3channel}")
    #     for key, val in color_class_map.items():
    #         color = eval(key)
    #         # color classes according to mapping
    #         # https://stackoverflow.com/questions/6483489/change-specific-rgb-color-pixels-to-another-color-in-image-file
    #         r1, g1, b1 = val, val, val # Original value
    #         r2, g2, b2 = color # Value that we want to replace it with
    #         red, green, blue = segmap_3channel[:,:,0], segmap_3channel[:,:,1], segmap_3channel[:,:,2]
    #         mask = (red == r1) & (green == g1) & (blue == b1)
    #         segmap_3channel[:,:,:3][mask] = [r2, g2, b2]
    #     return segmap_3channel

    # https://stackoverflow.com/questions/46103731/is-there-a-simple-method-to-highlight-the-mask/46105196
    def overlay_mask(
        self,
        bg_image,
        mask_image,
        *,
        alpha=0.5,
        fill=True,
        force_color=None,
        line_thickness=1,
    ):
        """Overlays a mask onto a background image

        Args:
            bg_image (str / image): image or path
            mask_image (str / image): cv2 image or path
            force_color ((int, int, int)): override default color of mask
        """
        # bg copy for background
        result_image = opencv_utils.get_image(
            bg_image
        ).copy()  # ensure original is not overwritten
        # bg copy for overlay
        overlay = opencv_utils.get_image(
            bg_image
        ).copy()  # ensure original is not overwritten
        # mask image for finding contours
        mask_image = opencv_utils.get_image(mask_image)
        mask_objects = self.find_image_objects(mask_image)

        if len(mask_objects) < 1:
            return result_image

        # draw the found contours on the overlay image
        for idx, o in enumerate(mask_objects):
            draw_color = (0, 0, 0)
            if force_color:
                draw_color = self.to_bgr(force_color)
            else:
                draw_color = self.to_bgr(o["color"])
            # draw
            cv2.drawContours(
                overlay, [o["contour"]], -1, draw_color, thickness=line_thickness
            )
            if fill:
                cv2.fillPoly(overlay, pts=[o["contour"]], color=draw_color)

        # apply the overlay to backround  image
        cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)

        return result_image

    def to_bgr(self, color_value):
        return tuple(reversed(color_value))

    def debug_annotation(self, image_path):
        cv_image = cv2.imread(image_path)
        height = cv_image.shape[0]
        width = cv_image.shape[1]
        cv2.imshow(f"original: {image_path}", cv_image)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        cv2.imshow(f"lab: {image_path}", lab)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f"gray: {image_path}", gray)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cv2.imshow(f"tresh: {image_path}", thresh)
        cnts, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.waitKey(0)
        for i, c in enumerate(cnts):
            approx = cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True)
            label = self.label(lab, approx)
            mask = np.zeros(lab.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            cv2.imshow(f"annot {i}, label {label}", mask)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_image_objects(self, bgr_image_or_path, use_bgr=False):
        """Extracts labels, colors, bounding boxes from an image.
        input: OpenCV image or path
        return: array of dicts({label, color, box})
        """
        cv_image = opencv_utils.get_image(bgr_image_or_path)
        height = cv_image.shape[0]
        width = cv_image.shape[1]

        # find contours
        # https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html

        # blur the resized image slightly, then convert it to both
        # blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

        # grayscale and the L*a*b* color spaces
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # find contours in the thresholded image (ONLY use external here the others will break finding labels!!)
        cnts, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
        # cnts = imutils.grab_contours(cnts)

        # loop over the contours
        # {box, color}
        results = []
        for c in cnts:

            # approximation: approximate polygonic curves with specific precision
            # https://www.youtube.com/watch?v=mVWQNeY1Pb4
            # accuracy: 0.001
            approx = cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True)

            # detect the shape of the contour and label the color
            label = self.label(lab, approx)
            color_value = self.label_to_color(label)
            if use_bgr:
                color_value = self.to_bgr(color_value)
            # box: tuple (x,y,w,h), color: tuple (r, g, b)
            results.append(
                {
                    "label": label,
                    "color": color_value,
                    "height": height,
                    "width": width,
                    "box": cv2.boundingRect(approx),
                    "contour": approx,
                }
            )
        del cv_image  # make sure to free memory
        return results

    def get_classes(self, image_or_path):
        res = self.find_image_objects(image_or_path)
        all_labels = [x["label"] for x in res]
        return all_labels

    def has_valid_objects(self, bgr_image_or_path):
        """Checks whether a mask image contains any objects and if those are correctly classified, e.g. not as 'background'

        Args:
            image_or_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        allowed_labels = self.get_labels(include_bg=False)
        classes = self.get_classes(bgr_image_or_path)
        found_invalid_classes = [x for x in classes if x not in allowed_labels]
        invalid_obj_found = len(found_invalid_classes) > 0
        has_objects = len(classes) > 0

        return has_objects and not invalid_obj_found

    def get_dominant_class(self, bgr_image_or_path):
        """Gets dominant class from input mask image.
           WARNING: if an image is given, make sure it is a cv2 image in BGR format,
                    otherwise, the predicted class will be wrong.

        Args:
            image_or_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        # old: count frequency of classes
        # all_labels = self.get_classes(image_path)
        # dominant_class =  utils.find_most_frequent(all_labels)

        # new: judge dominance by area
        objs = self.find_image_objects(bgr_image_or_path)
        class_area_totals = {}
        dominant_class = None
        max_area = 0
        for o in objs:
            label = o["label"]
            box = o["box"]
            box_area = box[2] * box[3]
            utils.increment_dict_key(class_area_totals, label, box_area)
            if class_area_totals[label] > max_area:
                max_area = class_area_totals[label]
                dominant_class = label

        return dominant_class

    def label(self, cv_lab_image, contour):
        """Label single countour in lab image"""
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(cv_lab_image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(cv_lab_image, mask=mask)[:3]

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
        if minDist[1] is not None:
            return self.colorNames[minDist[1]]
        else:
            return None
