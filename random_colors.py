#!/usr/bin/env python3.5
# from https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
from typing import Iterable, Tuple
import colorsys
import itertools
from fractions import Fraction
# from pprint import pprint

# HOW TO:
# get_colors("rgb"|"hsv"|"css", number)
# color conversion: see helper methods below


def zenos_dichotomy() -> Iterable[Fraction]:
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1, 2**k)


def fracs() -> Iterable[Fraction]:
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator  # [1,2,4,8,16,...]
        for j in range(1, i, 2):
            yield Fraction(j, i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)


HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]


def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6, 10)]:  # optionally use range
        for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
            yield (h, s, v)  # use bias for v here if you use range


def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))


flatten = itertools.chain.from_iterable


def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))


def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())


def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y * 255), x)
    return "rgb({},{},{})".format(*uint8tuple)


def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())


def get_colors(type, number):
    # sample 100 colors in css format
    # sample_colors = list(itertools.islice(css_colors(), 100))
    # pprint(sample_colors)
    if type == "rgb":
        return list(itertools.islice(rgbs(), number))
    elif type == "css":
        return list(itertools.islice(css_colors(), number))
    elif type == "hsv":
        return list(itertools.islice(hsvs(), number))


# http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/


def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor


def CSSToRGB(css_color):
    """ convert an 'rgb(R,G,B)' css string to an (R, G, B) tuple """
    only_numbers = css_color.replace("rgb(", "").replace(")", "")
    parts = only_numbers.split(",")
    return parts[0], parts[1], parts[2]


def HTMLColorToRGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple """
    colorstring = colorstring.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]
    if len(colorstring) != 6:
        raise ValueError("input #%s is not in #RRGGBB format" % colorstring)
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


def HTMLColorToPILColor(colorstring):
    """ converts #RRGGBB to PIL-compatible integers"""
    colorstring = colorstring.strip()
    while colorstring[0] == '#':
        colorstring = colorstring[1:]
    # get bytes in reverse order to deal with PIL quirk
    colorstring = colorstring[-2:] + colorstring[2:4] + colorstring[:2]
    # finally, make it numeric
    color = int(colorstring, 16)
    return color


def PILColorToRGB(pil_color):
    """ convert a PIL-compatible integer into an (r, g, b) tuple """
    hexstr = '%06x' % pil_color
    # reverse byte order
    r, g, b = hexstr[4:], hexstr[2:4], hexstr[:2]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


def PILColorToHTMLColor(pil_integer):
    return RGBToHTMLColor(PILColorToRGB(pil_integer))


def RGBToPILColor(rgb_tuple):
    return HTMLColorToPILColor(RGBToHTMLColor(rgb_tuple))
