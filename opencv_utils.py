#!/usr/bin/env python

###
# File: opencv_utils.py
# Created: Monday, 22nd June 2020 8:14:46 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:13:49 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###

# Utilities using OpenCV functions. Working OpenCV installation required.

import sys

import cv2
import numpy as np
from tqdm import tqdm

from . import utils

BLEND_ALPHA = 0.5
NUM_CHANNELS = 3
BB_COLOR = (255, 255, 255)

# Opencv Keycodes
KEY_ESCAPE = 27
KEY_ENTER = 13
KEY_SPACE = 32

# all opencv keycodes on different systems
WIN_ARROW_KEY_CODES = {
    "up": 2490368,
    "down": 2621440,
    "left": 2424832,
    "right": 2555904,
}
MAC_ARROW_KEY_CODES = {"up": 63232, "down": 63233, "left": 63234, "right": 63235}
LINUX_ARROW_KEY_CODES = {"up": 65362, "down": 65364, "left": 65361, "right": 65363}


def get_options_txt_image(img, options, selected, msg=None):
    if selected < 0 or selected > len(options) - 1:
        selected = 0
    user_prompt = ""
    if msg is not None:
        user_prompt += f"{msg}\n"

    for i, o in enumerate(options):
        if selected == i:
            user_prompt += f"{i}. [{o}]\n"
        else:
            user_prompt += f"{i}. {o}\n"

    user_prompt += f"\nHint: Use 'arrow' keys or 'w'/'s' to select options\nand 'Enter'/'Space' to confirm. Press 'Escape' to cancel."
    res = overlay_text(img, user_prompt, scale=0.5, y_pos=20)
    return res


def is_arrow_key_pressed(direction, pressed_keycode):
    """Checks if pressed key matches up/down/left/right key on any system.

    Args:
        direction (string): desired direction ("up", "down", "left", "right")
        pressed_keycode (integer): the keycode as recorded by OpenCV's waitKeyEx

    Returns:
        boolean: True if matching, false if not.
    """
    if direction not in WIN_ARROW_KEY_CODES.keys():
        return False
    return (
        pressed_keycode == WIN_ARROW_KEY_CODES[direction]
        or pressed_keycode == MAC_ARROW_KEY_CODES[direction]
        or pressed_keycode == LINUX_ARROW_KEY_CODES[direction]
    )


def gui_select_option(
    options, bg_image, *, window_title="Option Select", msg=None, default=0
):
    """
    Ask user to select one of several options using .
    Parameters
    ----------
    bg_image: string or np.array
        background image (path or image)
    window: string
        the window on which to show the select
    options: list of strings
        options to select
    msg: string
        user prompt message
    default: integer
        default selected idx
    return: tuple (integer, object)
        idx and value of selected option

    """
    bg = get_image(bg_image)
    bg_dims = get_img_dimensions(bg)
    overlay = create_blank_image(bg_dims["width"], bg_dims["height"])

    # user prompt
    sel_idx = default
    sel_option = None
    key = -1
    while key != KEY_SPACE and key != KEY_ENTER:
        prompt_img = get_options_txt_image(overlay, options, sel_idx, msg=msg)
        display_img = overlay_image(bg, prompt_img)
        cv2.imshow(window_title, display_img)
        # wait and listen to keypresses
        key = cv2.waitKeyEx(0)  # & 0xFF -> don't use here, disables arrow key use
        # use for finding out platform specific keycodes
        if key == ord("w") or is_arrow_key_pressed("up", key):
            sel_idx -= 1
            sel_idx %= len(options)
        elif key == ord("s") or is_arrow_key_pressed("down", key):
            sel_idx += 1
            sel_idx %= len(options)
        elif key == ord("q") or key == KEY_ESCAPE:
            # reset selection
            tqdm.write("Aborted...")
            sel_idx = -1
            sel_option = None
            break
        # update selected value
        sel_option = options[sel_idx]

    return sel_idx, sel_option


def is_image(variable):
    # openCV images are numpy arrays (TODO: more thorough check)
    return type(variable) is np.ndarray


def get_image(path_or_image):
    """Returns an OpenCV image, whether a path or an image is provided.
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
    if is_image(path_or_image):
        return path_or_image
    else:
        # path must have been provided (TODO: error handling)
        return cv2.imread(path_or_image)


def image_to_binary_image(img_or_path):
    """Converts 3 channel image into 1 channel binary image

    Args:
        img_or_path ([type]): [description]
    """
    img_or_path = get_image(img_or_path)
    img_copy = img_or_path.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  # to gray
    ret, img_copy = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY)  # thresholded
    contours = cv2.findContours(img_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    dims = get_img_dimensions(img_or_path)
    image_binary = np.zeros((dims["height"], dims["width"], 1), np.uint8)
    cv2.drawContours(
        image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1
    )
    return image_binary


def pil_to_cv_img(pil_image):
    """Converts PIL Image to OpenCV image.

    Args:
        pil_image (PIL Image): Input PIL image in RGB.

    Returns:
        [type]: OpenCV image in BGR.
    """
    opencv_img = np.array(pil_image)
    # Convert RGB to BGR
    opencv_img = opencv_img[:, :, ::-1].copy()
    return opencv_img


def get_bgr_image(img_or_path):
    image = get_image(img_or_path)
    img_copy = image.copy()
    return cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)


def get_rgb_image(img_or_path):
    image = get_image(img_or_path)
    img_copy = image.copy()
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)


def get_unique_color_values(img_or_path, axis=0):
    img_or_path = get_image(img_or_path)
    return np.unique(img_or_path.reshape(-1, img_or_path.shape[2]), axis=0)


# def cut_roi_from_image(foreground, background, mask):
#     result = np.zeros_like(foreground)
#     result[mask] = foreground[mask]
#     inv_mask = np.logical_not(mask)
#     result[inv_mask] = background[inv_mask]
#     return result


def get_img_dimensions(img):
    """Get image dimensions in form of a dictionary.
    Parameters
    ----------
    img: np.array
        input image
    return: dict of numbers
        {"width": img_width, "height": img_height, "channels": num_channels}
    """
    img = get_image(img)
    height = None
    width = None
    channels = -1
    if len(img.shape) == 2:
        # image is grayscale (1 channel)
        height, width = img.shape
        channels = 1
    else:
        # image has more than 1 channel
        height, width, channels = img.shape
    return {"width": width, "height": height, "channels": channels}


def create_blank_image(width, height, num_channels=NUM_CHANNELS):
    """Creates a blank (black) image

    Args:
        width (integer): the width
        height (integer): the height
        num_channels (integer, optional): Number of channels. Defaults to NUM_CHANNELS.

    Returns:
        np.array: A blank black image of size width x height.
    """
    if num_channels == 1:
        return np.zeros(shape=(height, width), dtype=np.uint8)
    else:
        return np.zeros(shape=(height, width, num_channels), dtype=np.uint8)


def draw_rectangle(img, bb, color=BB_COLOR, thickness=cv2.FILLED):
    """Draws a rectangle to an image in a desired color (default: white)
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
    cv2.rectangle(img, (bb[0], bb[1]), (x2, y2), color=color, thickness=thickness)


def draw_line(img, p_a, p_b, *, color=(255, 255, 255), thickness=1):
    """Draws a line from p_a to p_b

    Args:
        image ([type]): path or image
        p_a (tuple of int): point in form (x, y)
        p_b (tuple of int): point in form (x, y)
        color (tuple, optional): [description]. Defaults to (255, 255, 255).
        thickness (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    img = get_image(img)
    img_altered = img.copy()
    cv2.line(img_altered, p_a, p_b, color, thickness)
    return img_altered


def draw_horizontal_line(img, x_pos_percent=0.5, line_thickness=1, color=(0, 255, 0)):
    """Draws a horizontal line through an input picture. position is given as a percentage.

    Args:
        img ([type]): path or image
        x_pos_percent ([type]): percentage of width, default 0.50
        color (tuple, optional): [description]. Defaults to (0,255,0).
    """
    img = get_image(img)
    img_altered = img.copy()
    width, height = img.shape[1], img.shape[0]
    x1, y1 = int(width * x_pos_percent), 0
    x2, y2 = int(width * x_pos_percent), height
    cv2.line(img_altered, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
    return img_altered


def convert_to_rgba(img_or_path):
    """Converts an image with potentially 1/3 channels to an rgba image. Leaves rgba or any other images (2, 4, 4+ channels) as they are.

    Args:
        img_or_path: path or image
    """
    img = get_image(img_or_path)
    dims = get_img_dimensions(img)
    if dims["channels"] == 1:
        return cv2.cvtColor(img_or_path, cv2.COLOR_GRAY2BGRA)
    elif dims["channels"] == 3:
        return cv2.cvtColor(img_or_path, cv2.COLOR_BGR2BGRA)
    else:
        return img


def concatenate_images(img1, img2, axis=1):
    """Concatanates two images horizontally (axis=1, default) or vertically(axis=0).
    INPORTANT: outputs BGRA image, convert if other format needed!
    Parameters
    ----------
    img1: path or image
    img2: path or image
    """
    img1 = get_image(img1)
    img2 = get_image(img2)
    # ensure that dimensions match, by converting all imgs to RGBA
    img1 = convert_to_rgba(img1)
    img2 = convert_to_rgba(img2)
    return np.concatenate((img1, img2), axis=axis)


# src: https://stackoverflow.com/questions/40527769/removing-black-background-and-make-transparent-from-grabcut-output-in-python-opencv-python
def get_transparent_img(img):
    """Makes black color (0,0,0) in an img transparent.
    Parameters
    ----------
    img: image or path
    """
    img = get_image(img)

    # if image already has alpha channel convert it to BGR temporarily
    dims = get_img_dimensions(img)
    if dims["channels"] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    bgra = [b, g, r, alpha]
    dst = cv2.merge(bgra, 4)

    return dst


def show_image(
    image, title="image", pos=None, destroy_after_keypress=True, enable_keypress=True
):
    """Shows image with option to set position and enabling ESCAPE to quit.
    Parameters:
    -----------
    image: object or path
        OpenCv image
    title: string
        window title
    pos: dict of integers (WARNING: values of 0 can behave funny - best use e.g. 10 instead)
        {"x": x_pos, "y": y_pos}
    """
    image = get_image(image)
    cv2.imshow(title, image)
    if pos is not None:
        cv2.moveWindow(title, pos["x"], pos["y"])
    if enable_keypress:
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit()
        if destroy_after_keypress:
            cv2.destroyWindow(title)  # cleanup


def RGB_to_BGR(rgb):
    r, g, b = rgb
    return b, g, r


def add_colored_border(img_or_path,*, color_in_rgb=(255, 255, 255), size_pixel=5):
    img = get_image(img_or_path)
    color_bgr = RGB_to_BGR(color_in_rgb)
    img_bordered = cv2.copyMakeBorder(img, size_pixel, size_pixel, size_pixel, size_pixel, cv2.BORDER_CONSTANT, value=color_bgr)
    return img_bordered


def overlay_text(
    img,
    txt,
    *,
    x_pos=10,
    y_pos=25,
    scale=1,
    color_in_rgb=(255, 255, 255),
    color_mix=None,
    thickness=1,
):
    """Overlays text on an opencv image, does not change original image. Supports multiline text using '\n'.
    Parameters
    ----------
    img: path or opencv image
    txt: string
    color_mix: list of RGB color tuples overriding 'color' (each line is created in a different color out of the mix)
    Returns: np.array
        image with a text overlay
    """
    img = get_image(img)
    altered_img = img.copy()
    # height, _, _ = altered_img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x_pos, y_pos)
    fontScale = scale
    fontColors = [color_in_rgb]
    lineType = cv2.FILLED

    if color_mix is not None and len(color_mix) > 0:
        fontColors = color_mix

    for i, line in enumerate(txt.split("\n")):
        cur_color_idx = i % len(fontColors)
        cur_color = fontColors[cur_color_idx]
        cur_color = RGB_to_BGR(cur_color)
        cv2.putText(
            img=altered_img,
            text=line,
            org=bottomLeftCornerOfText,
            fontFace=font,
            fontScale=fontScale,
            color=cur_color,
            lineType=lineType,
            thickness=thickness,
        )
        size, _ = cv2.getTextSize(txt, font, fontScale, lineType)
        # baseline += thickness
        w, h = size
        x_pos, y_pos = bottomLeftCornerOfText
        bottomLeftCornerOfText = (x_pos, y_pos + h + 5 + thickness)

    return altered_img


# src: https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e
# src2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
def overlay_image(background_img, img_to_overlay, x=0, y=0, overlay_size=None):
    """Overlays a potentially transparant PNG onto another image using CV2

    Args:
        background_img (Union[string, np.array]): The background image or path
        img_to_overlay ([type]): The image/path to overlay (is converted to be transparent if needed via 'get_transparent_img')
        x ([integer]): x location to place the top-left corner of our overlay
        y ([integer]): y location to place the top-left corner of our overlay
        overlay_size ([type], optional): The size to scale our overlay to (tuple), no scaling if None. Defaults to None.

    Returns:
        [np.array]: Background image with overlay on top
    """
    background_img = get_image(background_img)
    img_to_overlay = get_transparent_img(img_to_overlay)

    bg_img = background_img.copy()
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)  # ensure bg picture is RGB only

    if overlay_size is not None:
        img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    # mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y : y + h, x : x + w]

    # Black-out the area behind the overlay in our original ROI
    # img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy())

    # Mask out the overlay from the overlay image.
    # img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color)

    # Update the original image with the new ROI
    bg_img[y : y + h, x : x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


def rgb_to_bgr_image(in_rgb):
    return cv2.cvtColor(in_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_image(in_rgb):
    return cv2.cvtColor(in_rgb, cv2.COLOR_BGR2RGB)


# maybe better: https://stackoverflow.com/questions/51365126/combine-2-images-with-mask
def blend_image(in_bg, in_overlay, blend=BLEND_ALPHA):
    """Blends two images with transparency.

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
    img_bg = get_image(in_bg)
    img_overlay = get_image(in_overlay)

    # copy images as to no alter originals
    img_bg_copy = img_bg.copy()
    img_overlay_copy = img_overlay.copy()

    img_bg_copy = cv2.cvtColor(img_bg_copy, cv2.COLOR_BGR2BGRA)
    img_overlay_bgra = get_transparent_img(img_overlay_copy)
    # blend_images
    beta = 1.0 - blend
    result = cv2.addWeighted(img_bg_copy, blend, img_overlay_bgra, beta, 0.0)
    return result


def blend_images(img, img_overlay_array, blend=BLEND_ALPHA):
    """Blends an array of images with a bg image using transparency.

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
    if not isinstance(img_overlay_array, list):
        tqdm.write("blend_images needs an array as input")
        return None
    img = get_image(img)
    res = img
    for o in img_overlay_array:
        o = get_image(o)
        res = blend_image(res, o, blend)
    return res


def resize_image(img_or_path, w, h, interpolation=cv2.INTER_AREA):
    img = get_image(img_or_path)
    dim = (w, h)
    resized_img = cv2.resize(img, dim, interpolation)
    return resized_img


def scale_image(img_or_path, scale_factor, interpolation=cv2.INTER_AREA):
    img = get_image(img_or_path)
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation)
    return resized_img


def get_bbox_center(box):
    """Calculate center of a bounding box of type: tuple (x, y, w, h)"""
    cx = int(box[0] + 0.5 * box[2])
    cy = int(box[1] + 0.5 * box[3])
    return {"x": cx, "y": cy}


def get_contour_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return {"x": cx, "y": cy}


def move_contour_to(contour, center_x, center_y):
    """Moves a contour by its centroid, given coords are taken as new position center coordinates

    Args:
        contour ([type]): [description]
        center_x ([type]): [description]
        center_y ([type]): [description]
    """
    c_centroid = get_contour_centroid(contour)
    new_x = center_x - c_centroid["x"]
    new_y = center_y - c_centroid["y"]
    new_contour = contour + [new_x, new_y]
    # tqdm.write(f"o {c_centroid['x']}, {c_centroid['y']}")
    # tqdm.write(f"t {center_x}, {center_y}")
    # tqdm.write(f"move {new_x}, {new_y}")
    return new_contour


def save_image(img, file_path, info=True):
    # Saving the image
    cv2.imwrite(file_path, img)
    if info:
        tqdm.write(f"Wrote {file_path}")
