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
import math
from PIL import Image


# colors in opencv BGR
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

BLEND_ALPHA = 0.5
NUM_CHANNELS = 3
BB_COLOR = COLOR_WHITE


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

# misc
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
def get_text_size(text, font=TEXT_FONT, scale: float = 1, thickness=1):
    lines = text.split("\n")
    num_rows = len(lines)
    longest_string = max(lines, key=len)

    # get boundary of text
    textsize = cv2.getTextSize(longest_string, font, scale, thickness)[0]
    return {"width": textsize[0], "height": textsize[1] * num_rows}


def get_options_txt_image(
    img,
    options,
    selected,
    msg=None,
    pos="top",
    draw_outline=False,
    show_help=True,
    show_idx=True,
    scale=0.5,
    thickness=1,
):
    """Creates an image with text options (used by gui_select_option)

    Args:
        img (_type_): _description_
        options (_type_): _description_
        selected (_type_): _description_
        msg (_type_, optional): _description_. Defaults to None.
        pos (str, optional): "top", "bottom" or "center". Defaults to "top".
        draw_outline: optionally draw a black outline around the text
        show_help: show user info for selecting values
        show_idx: show available indices for selecting values

    Returns:
        _type_: _description_
    """

    if selected < 0 or selected > len(options) - 1:
        selected = 0
    user_prompt = ""
    if msg is not None:
        user_prompt += f"{msg}\n"

    for i, o in enumerate(options):
        idx_txt = ""
        if show_idx:
            idx_txt = f"{i}. "
        if selected == i:
            user_prompt += f"{idx_txt}[{o}]\n"
        else:
            user_prompt += f"{idx_txt}{o}\n"

    if show_help:
        user_prompt += f"\nHint: Use 'arrow' keys or 'w'/'s' to select options\nand 'Enter'/'Space' to confirm. Press 'Escape' to cancel."

    # color
    ol_color = None
    if draw_outline:
        ol_color = COLOR_BLACK

    # positions

    # top
    x_pos = 10
    y_pos = 20
    img_dims = get_img_dimensions(img)
    if pos == "bottom":
        y_pos = img_dims["height"] - 40
    elif pos == "center":
        text_size = get_text_size(user_prompt, scale=scale, thickness=thickness)
        # get center coords based on boundary
        x_pos = (img_dims["width"] - text_size["width"]) / 2
        y_pos = (img_dims["height"] - text_size["height"]) / 2

    if x_pos < 0:
        x_pos = 0
    if x_pos > img_dims["width"]:
        x_pos = img_dims["width"]
    if y_pos < 0:
        y_pos = 0
    if y_pos > img_dims["height"]:
        y_pos = img_dims["height"]

    x_pos = int(x_pos)
    y_pos = int(y_pos)

    res = overlay_text(
        img,
        user_prompt,
        scale=scale,
        thickness=thickness,
        x_pos=x_pos,
        y_pos=y_pos,
        outline_color=ol_color,
    )

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
    options,
    bg_image,
    *,
    window_title="Option Select",
    msg=None,
    default=0,
    position="top",
    draw_outline=False,
    show_help=True,
    show_idx=True,
    scale=0.5,
    thickness=1,
):
    """
    Ask user to select one of several options using a string list.
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
    show_help:
        shows help message for selecting options
    show_idx:
        shows option indices next to options
    default: integer
        default selected idx
    position: display position ["top", "bottom"]
    draw_outline: draws outline around text
    return: tuple (integer, object)
        idx and value of selected option

    """
    bg = get_image(bg_image)
    bg_dims = get_img_dimensions(bg)

    # old method - create overlay image with text and combine with bg
    # overlay = create_blank_image(bg_dims["width"], bg_dims["height"])

    # user prompt
    sel_idx = default
    sel_option = None
    key = -1
    while key != KEY_SPACE and key != KEY_ENTER:
        display_img = bg.copy()
        display_img = get_options_txt_image(
            display_img,
            options,
            sel_idx,
            msg=msg,
            pos=position,
            draw_outline=draw_outline,
            show_help=show_help,
            show_idx=show_idx,
            scale=scale,
            thickness=thickness,
        )
        # display_img = overlay_image(bg, prompt_img) # old method
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
        del display_img

    return sel_idx, sel_option


def is_image(variable):
    # openCV images are numpy arrays (TODO: more thorough check)
    return type(variable) is np.ndarray


def get_image(path_or_image, copy=False):
    """Returns an OpenCV image, whether a path or an image is provided.
    Ensures that most methods can be used by passing the image path OR the image itself.
    Parameters:
    -----------
    path_or_image: string or openCV image object
        path or OpenCV image
    copy: create a copy of potentially already loaded opencv image
    returns:
        loaded OpenCV image
        (path_or_image if it already is an OpenCV image,
         a newly loaded one otherwise)

    """
    if is_image(path_or_image):
        if copy:
            return path_or_image.copy()
        else:
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
        image_binary, [max(contours, key=cv2.contourArea)], -1, COLOR_WHITE, -1
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
    img_copy = get_image(img_or_path, True)
    return cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)


def get_rgb_image(img_or_path):
    img_copy = get_image(img_or_path, True)
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


def shift_image(img_or_path, x_shift, y_shift):
    img_alt = get_image(img_or_path, True)
    img_dims = get_img_dimensions(img_alt)
    # src: https://www.youtube.com/watch?v=FWg2BPmXvdk
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]]) # type: ignore
    return cv2.warpAffine(img_alt, matrix, (img_dims["width"], img_dims["height"]))


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


def draw_polygon(img, poly,*, is_closed=True, color=BB_COLOR, thickness=cv2.FILLED):
    """Draws a polygon to an image in a desired color (default: white)
    Parameters
    ----------
    img: np array (Opencv image) or path
    poly: array / list of coords -> [[x1, y1], [x2, y2], ...]
        bounding box of format (x, y, w, h)
    color: 3-tuple
        RGB color values
    """
    img = get_image(img)
    pts_2d = np.array(poly, np.int32)
    pts_1d = pts_2d.reshape((-1, 1, 2)) # pts to 1d points
    if thickness == cv2.FILLED:
        cv2.fillPoly(img, pts=[pts_2d], color=color)
    else:
        cv2.polylines(img, [pts_1d], isClosed=is_closed, thickness=thickness, color=color)


def draw_partial_circle(
    image,
    *,
    copy_image=True,
    center=(0, 0),
    radius=10,
    start_angle=180,
    end_angle=360,
    color=COLOR_WHITE,
    thickness=1,
):
    """Draws a (partial) circle around point by taking a starting and ending angle.

    Args:
        image (_type_): _description_
        radius (_type_): _description_
        start_angle (int, optional): _description_. Defaults to 180.
        end_angle (int, optional): _description_. Defaults to 360.
        thickness (int, optional): _description_. Defaults to 1.
    """
    img = get_image(image, copy_image)
    # Ellipse parameters
    axes = (radius, radius)
    angle = 0
    # http://docs.opencv.org/modules/core/doc/drawing_functions.html#ellipse
    cv2.ellipse(img, center, axes, angle, start_angle, end_angle, color, thickness)
    return img


def draw_line(img, p_a, p_b, *, create_copy=True, color=(255, 255, 255), thickness=1):
    """Draws a line from p_a to p_b

    Args:
        image ([type]): path or image
        p_a (tuple of int): point in form (x, y)
        p_b (tuple of int): point in form (x, y)
        create_copy: return altered image copy or alter original image
        color (tuple, optional): [description]. Defaults to COLOR_WHITE.
        thickness (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    img = get_image(img, create_copy)
    cv2.line(img, p_a, p_b, color, thickness)
    return img


def draw_rotated_line(
    img,
    x_pos_percent=0.5,
    y_pos_percent=0.5,
    angle=0,
    line_thickness=1,
    color=COLOR_GREEN,
):
    """Draws a straigh and rotated line through an input picture (always touching borders). rot point is given by percentages.

    Args:
        img ([type]): path or image
        x_pos_percent ([type]): percentage of width, default 0.50
        y_pos_percent ([type]): percentage of height, default 0.50
        angle ([type]): rotation angle, default 0, -180 < 0 < 180 (horizontal line)
        color (tuple, optional): [description]. Defaults to (0,255,0).
    """
    img_altered = get_image(img, True)
    width, height = img.shape[1], img.shape[0]

    # point around which to rotate
    rot_pt_x, rot_pt_y = int(width * x_pos_percent), int(height * y_pos_percent)

    # horizontal line through rot point (default)
    x1, y1 = 0, rot_pt_y
    x2, y2 = width, rot_pt_y

    if angle == 180:
        angle = 0
    if angle == -180:
        angle = 0

    if angle != 0:
        x1_length = (rot_pt_x - width) / math.cos(angle)
        y1_length = (rot_pt_y - height) / math.sin(angle)
        length = max(abs(x1_length), abs(y1_length))
        x1 = rot_pt_x + length * math.cos(math.radians(angle))
        y1 = rot_pt_y + length * math.sin(math.radians(angle))

        x2_length = (rot_pt_x - width) / math.cos(angle + 180)
        y2_length = (rot_pt_y - height) / math.sin(angle + 180)
        length = max(abs(x2_length), abs(y2_length))
        x2 = rot_pt_x + length * math.cos(math.radians(angle + 180))
        y2 = rot_pt_y + length * math.sin(math.radians(angle + 180))

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

    cv2.line(img_altered, (x1, y1), (x2, y2), COLOR_GREEN, thickness=line_thickness)
    return img_altered


def draw_horizontal_line(img, y_pos_percent=0.5, line_thickness=1, color=COLOR_GREEN):
    """Draws a horizontal line through an input picture. position is given as a percentage.

    Args:
        img ([type]): path or image
        y_pos_percent ([type]): percentage of width, default 0.50
        color (tuple, optional): [description]. Defaults to (0,255,0).
    """
    img_altered = get_image(img, True)
    width, height = img.shape[1], img.shape[0]
    x1, y1 = 0, int(height * y_pos_percent)
    x2, y2 = width, int(width * y_pos_percent)
    cv2.line(img_altered, (x1, y1), (x2, y2), COLOR_GREEN, thickness=line_thickness)
    return img_altered


def draw_vertical_line(img, x_pos_percent=0.5, line_thickness=1, color=COLOR_GREEN):
    """Draws a vertical line through an input picture. position is given as a percentage.

    Args:
        img ([type]): path or image
        x_pos_percent ([type]): percentage of width, default 0.50
        color (tuple, optional): [description]. Defaults to (0,255,0).
    """
    img_altered = get_image(img, True)
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


def add_colored_border(img_or_path, *, color=COLOR_WHITE, size=5):
    """[summary]

    Args:
        img_or_path ([type]): Image of Path.
        color (tuple, optional): RGB color tuple. Defaults to white.
        size_pixel (int, optional): Border size in pixels. Defaults to 5.

    Returns:
        [type]: bordered cv2 image
    """
    img = get_image(img_or_path)
    color_bgr = RGB_to_BGR(color)
    img_bordered = cv2.copyMakeBorder(
        img,
        size,
        size,
        size,
        size,
        cv2.BORDER_CONSTANT,
        value=color_bgr,
    )
    return img_bordered


def overlay_text(
    img,
    txt,
    *,
    x_pos=10,
    y_pos=25,
    scale: float = 1,
    color=COLOR_WHITE,
    color_mix=None,
    thickness=1,
    outline_color=None,
):
    """Overlays text on an opencv image, does not change original image. Supports multiline text using '\\n'.
    Parameters
    ----------
    img: path or opencv image
    txt: string
    color: RGB Tuple
    color_mix: list of RGB color tuples overriding 'color' (each line is created in a different color out of the mix)
    draw_outline: draw outline around text
    outline_color: RGB color tuple for outline - no outline is drawn if set to None (default)

    Known issue:
        - opencv uses the Hershey font used and this only supports a limited subset of ASCII characters (https://github.com/opencv/opencv/issues/11427)
        - a workaround could be using PIL:

            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            def print_utf8(image, text, color):
                fontName = 'FreeSerif.ttf'
                font = ImageFont.truetype(fontName, 18)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((0, image.shape[0] - 30), text, font=font,
                    fill=(color[0], color[1], color[2], 0))
                image = np.array(img_pil)
                return image

    Returns: np.array
        image with a text overlay
    """
    altered_img = get_image(img, True)
    # height, _, _ = altered_img.shape
    font = TEXT_FONT
    bottomLeftCornerOfText = (x_pos, y_pos)
    fontScale = scale
    fontColors = [color]
    lineType = cv2.FILLED

    rows = txt.split("\n")
    num_rows = len(rows)

    text_size = get_text_size(txt, scale=scale, thickness=thickness)
    row_height = int(text_size["height"] / num_rows)
    row_gap = 16

    if color_mix is not None and len(color_mix) > 0:
        fontColors = color_mix

    for i, line in enumerate(rows):
        # text outline
        if outline_color != None:
            ol_color = RGB_to_BGR(outline_color)
            cv2.putText(
                img=altered_img,
                text=line,
                org=bottomLeftCornerOfText,
                fontFace=font,
                fontScale=fontScale,
                color=ol_color,
                lineType=lineType,
                thickness=thickness * 2,
            )
        # text
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
        x_pos, y_pos = bottomLeftCornerOfText
        # bottomLeftCornerOfText = (x_pos, y_pos + h + 5 + thickness)
        bottomLeftCornerOfText = (x_pos, int(y_pos + row_height + row_gap * scale))

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


def rotate_image(image, angle):
    """Rotates image using angle.

    Args:
        image (_type_): _description_
        angle (float): angle - positive = clockwise, negative = counterclockwise

    Returns:
        _type_: _description_
    """
    angle = -angle  # invert direction (default positive dir = counterclockwise)
    image = get_image(image)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


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


def get_distance(point_a, point_b):
    """calcs distance between two points

    Args:
        point_a (int): tuple of x1, y1
        point_b (int): tuple of x2, y2

    Returns:
        float: distance
    """
    x_diff = point_a[0] - point_b[0]
    y_diff = point_a[1] - point_b[1]
    dist = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
    return dist


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


def is_window_visible(window_title):
    """Checks whether a window is visible.

    Args:
        window_title (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1
    except Exception as e:
        print(e)
        return False


def set_on_mouse_func(window_title, cb, params):
    """Sets a mouse callback function for listening to mouse events.

    Args:
        window_title (str): the window title as string
        cb (function): callback function including params (event, x, y, flags, params)
                        event (flag)    : OpenCV mouse event, common -> EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, EVENT_LBUTTONUP
                                          (all events: https://docs.opencv.org/4.x/d0/d90/group__highgui__window__flags.html#ga927593befdddc7e7013602bca9b079b0)

                        x (int)         : x coord

                        y (int)         : y coord

                        flags (flag)    : https://docs.opencv.org/4.x/d0/d90/group__highgui__window__flags.html#gaab4dc057947f70058c80626c9f1c25ce

                        params          : additional custom params
        params: additional custom params (gets passed into callback params, recommended to use dict)
    """
    # window must be visible, if not create an empty window
    if not is_window_visible(window_title):
        cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, cb, params)


def save_image(img, file_path, info=True):
    # Saving the image
    cv2.imwrite(file_path, img)
    if info:
        tqdm.write(f"Wrote {file_path}")


# IMAGE TRANSFORMATIONS

def opencv_to_pil(img_cv):
    # src: https://chowdera.com/2021/07/20210707185741848M.html
    # opencv -> pil
    return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))

def pil_to_opencv(img_pil):
    # src: https://chowdera.com/2021/07/20210707185741848M.html
    # pil -> opencv
    return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)

