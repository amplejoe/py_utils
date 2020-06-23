#!/usr/bin/env python

# Utilities using OpenCV functions. Working OpenCV installation required.

from . import utils
from cv2 import cv2
import numpy as np
import sys

BLEND_ALPHA = 0.5


def concatenate_images(img1, img2):
    """ Concatanates two images horizontally
    """
    # ensure that dimensions match, by converting all imgs to RGBA
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    return np.concatenate((img1, img2), axis=1)


def make_transparent(annot_img):
    """ Makes (black) annot bg transparent.
    """
    tmp = cv2.cvtColor(annot_img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(annot_img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst


def show_image(image, title, pos=None):
    """ Shows image with option to set position and enabling ESCAPE to quit.
        Parameters:
        -----------
        image: object
            OpenCv iamge
        title: string
            window title
        pos: dict of integers
            {"x": x_pos, "y": y_pos}
    """
    cv2.imshow(title, image)
    if pos is not None:
        cv2.moveWindow(title, pos["x"], pos["y"])
    key = cv2.waitKey(0)
    if key == 27:
        sys.exit()


def add_text_to_image(img, txt, x_pos=10, y_offset=10):
    """ Adds text to an opencv image.
    """
    height, width, channels = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x_pos, height - y_offset)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, txt,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


# maybe better: https://stackoverflow.com/questions/51365126/combine-2-images-with-mask
def overlay_image(img_bg_path, img_overlay_path):
    """ Overlays an image over another.

        Parameters:
        ----------

        img_bg: string
            path to background image
        img_overlay: string
            path to overlay image. All transparent areas must be black (0, 0, 0).
    """
    bg_img = cv2.imread(img_bg_path)
    img_overlay = cv2.imread(img_overlay_path)

    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
    img_overlay_rgba = make_transparent(img_overlay)
    # blend_images
    beta = (1.0 - BLEND_ALPHA)
    result = cv2.addWeighted(bg_img, BLEND_ALPHA, img_overlay_rgba, beta, 0.0)
    return result