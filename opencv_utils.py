#!/usr/bin/env python

# Utilities using OpenCV functions. Working OpenCV installation required.

from . import utils
from cv2 import cv2
import numpy as np
import sys

BLEND_ALPHA = 0.5
NUM_CHANNELS = 3
BB_COLOR = (255, 255, 255)

def get_image(path_or_image):
    """ Returns an OpenCV image, whether a path or an image is provided.
        Ensures that most methods can be used by passing the image path OR the image itself.
        Parameters:
        -----------
        path_or_image: string or openCV image object
            path or OpenCV image
        returns:
            loaded OpenCV image
            (path_or_image if it already is an OpenCV image, 
             a newly loaded one otherwise)

    """
    if type(path_or_image) is np.ndarray:
        # openCV images are numpy arrays (TODO: more thorough check)
        return path_or_image
    else:
        # path must have been provided (TODO: error handling)
        return cv2.imread(path_or_image)


def get_img_dimensions(img):
    """ Get image dimensions in form of a dictionary.
        Parameters
        ----------
        img: np.array
            input image
        return: dict of numbers
            {"width": img_width, "height": img_height, "channels": num_channels}
    """
    dims = img.shape
    return {"width": dims[0], "height": dims[1], "channels": dims[2]}


def create_blank_image(width, height, num_channels = NUM_CHANNELS):
    return np.zeros(shape=(width, height, num_channels), dtype=np.uint8)

def draw_rectangle(img, bb, color = BB_COLOR):
    """ Draws a rectangle to an image in a desired color (default: white)
        Parameters
        ----------
        img: np array (Opencv image) or path     
        bb: 4-tuple
            bounding box of format (x, y, w, h)
        color: 3-tuple
            RGB color values
    """
    img = get_image(img)
    x2 = bb[0] + bb[2] 
    y2 = bb[1] + bb[3]
    cv2.rectangle(img, (bb[0], bb[1]), (x2, y2), color=color, thickness=cv2.FILLED)
    return img


def concatenate_images(img1, img2):
    """ Concatanates two images horizontally
        Parameters
        ----------
        img1: path or image
        img2: path or image
    """
    img1 = get_image(img1)
    img2 = get_image(img2)
    # ensure that dimensions match, by converting all imgs to RGBA
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    return np.concatenate((img1, img2), axis=1)


def get_transparent_img(img):
    """ Makes img transparent.
        Parameters
        ----------
        img: image or path
    """
    img = get_image(img)
    dims = get_img_dimensions(img)
    dst = None
    # check if image has alpha channel or not
    if dims["channels"] < 4:
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img)
        bgra = [b, g, r, alpha]
        dst = cv2.merge(bgra, 4)
    else:
        dst = img
    return dst


def show_image(image, title, pos=None):
    """ Shows image with option to set position and enabling ESCAPE to quit.
        Parameters:
        -----------
        image: object or path
            OpenCv iamge
        title: string
            window title
        pos: dict of integers
            {"x": x_pos, "y": y_pos}
    """
    image = get_image(image)
    cv2.imshow(title, image)
    if pos is not None:
        cv2.moveWindow(title, pos["x"], pos["y"])
    key = cv2.waitKey(0)
    if key == 27:
        sys.exit()


def add_text_to_image(img, txt, x_pos=10, y_offset=10):
    """ Adds text to an opencv image.
        Parameters
        ----------
        img: path or opencv image
        txt: string
    """
    img = get_image(img)
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
    return img


# maybe better: https://stackoverflow.com/questions/51365126/combine-2-images-with-mask
def overlay_image(img_bg, img_overlay, blend = BLEND_ALPHA):
    """ Overlays an image over another.

        Parameters
        ----------

        img_bg: string or image
            background image (path or image itself)
        img_overlay: string or image
            overlay image (path or image itself)
            All transparent areas must be black (0, 0, 0)
        blend: float
            1.0 - 0.0 most to least amount of transparency applied
    """
    bg_img =  get_image(img_bg)
    img_overlay = get_image(img_overlay)

    # copy images as to no alter originals
    bg_img_copy = bg_img.copy()
    img_overlay_copy = img_overlay.copy()

    bg_img_copy = cv2.cvtColor(bg_img_copy, cv2.COLOR_BGR2BGRA)
    img_overlay_bgra = get_transparent_img(img_overlay_copy)
    # blend_images
    beta = (1.0 - blend)
    result = cv2.addWeighted(bg_img_copy, blend, img_overlay_bgra, beta, 0.0)
    return result


def overlay_images(img, img_overlay_array, blend = BLEND_ALPHA):
    """ Overlays an array of images over another.

        Parameters
        ----------

        img_bg: string or image
            background image (path or image itself)
        img_overlay: list of strings of images
            overlay images (paths or images themselves)
            All transparent areas must be black (0, 0, 0)
        blend: float
            1.0 - 0.0 most to least amount of transparency applied
    """
    img = get_image(img)
    res = img
    for o in img_overlay_array:
        o = get_image(o)
        res = overlay_image(res, o, blend)
    return res
