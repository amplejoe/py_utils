#!/usr/bin/env python

###
# File: random_colors.py
# Created: Tuesday, 12th May 2020 5:46:35 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:14:13 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###

# based on
# https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python

import colorsys
import itertools
from random import randint

# from pprint import pprint

# HOW TO:
# get_colors("rgb"|"hsv"|"css"|"hex", number)
# color conversion: see helper methods below


def get_colors(type, number):
    """Generate N random distinct colors.

    Args:
        type ([type]): 'rgb', 'hsv', 'css', 'hex'
        number ([type]): number of colors

    Returns:
        [type]: list of colors
    """
    colors = {k: [] for k in "rgb"}
    for i in range(number):
        temp = {k: randint(0, 255) for k in "rgb"}
        for k in temp:
            while 1:
                c = temp[k]
                t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 255)
                if t.intersection(colors[k]):
                    temp[k] = randint(0, 255)
                else:
                    break
            colors[k].append(temp[k])

    rgb_tuples = [
        (colors["r"][i], colors["g"][i], colors["b"][i]) for i in range(number)
    ]
    if type == "css":
        return [rgb_2_css(i) for i in rgb_tuples]
    elif type == "hex":
        return [rgb_2_hex(i) for i in rgb_tuples]
    elif type == "rgb":
        return rgb_tuples
    elif type == "hsv":
        return [colorsys.rgb_to_hsv(i[0], i[1], i[2]) for i in rgb_tuples]


# http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/


def css_2_rgb(css_color):
    """ convert an 'rgb(R,G,B)' css string to an (R, G, B) tuple """
    only_numbers = css_color.replace("rgb(", "").replace(")", "")
    parts = only_numbers.split(",")
    return int(parts[0]), int(parts[1]), int(parts[2])


def rgb_2_css(rgb_tuple):
    """ convert an (R, G, B) tuple into an 'rgb(R,G,B)' css string  """
    return f"rgb{str(rgb_tuple).replace(' ', '')}"


def rgb_2_hex(rgb_tuple):
    """ convert (R, G, B) tuple to #RRGGBB """
    return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])


def hex_2_rgb(hexcode):
    """ convert #RRGGBB to an (R, G, B) tuple """
    stripped = hexcode.lstrip("#")
    return tuple(int(stripped[i : i + 2], 16) for i in (0, 2, 4))


def hex_2_pil_color(colorstring):
    """ converts #RRGGBB to PIL-compatible integers"""
    colorstring = colorstring.strip()
    while colorstring[0] == "#":
        colorstring = colorstring[1:]
    # get bytes in reverse order to deal with PIL quirk
    colorstring = colorstring[-2:] + colorstring[2:4] + colorstring[:2]
    # finally, make it numeric
    color = int(colorstring, 16)
    return color


def pil_color_2_rgb(pil_color):
    """ convert a PIL-compatible integer into an (r, g, b) tuple """
    hexstr = "%06x" % pil_color
    # reverse byte order
    r, g, b = hexstr[4:], hexstr[2:4], hexstr[:2]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


def pil_color_2_hex(pil_integer):
    return rgb_2_hex(pil_color_2_rgb(pil_integer))


def rgb_2_pil_color(rgb_tuple):
    return hex_2_pil_color(rgb_2_hex(rgb_tuple))
