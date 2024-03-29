#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
__description__ = """
[DESCRIPTION]
"""
# =============================================================================
# @author   : [AUTHOR]
# @date     : [DATE]
# @version  : 1.0
# =============================================================================

import argparse
from py_utils import utils
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

IN_IMG_EXT = [".jpg", ".png"]
IN_VID_EXT = [".mp4", ".avi", ".mov"]
IN_DATA_EXT = [".npy", ".csv", ".json"]


def main():

    if (
        g_args.output == g_args.input
        or g_args.output == "."
        or g_args.output == g_args.script_dir
    ):
        exit("IN cannot be the same as OUT, '.' or script path.")

    # if not (utils.confirm_overwrite(g_args.output, "n")):
    #     exit("Aborted folder creation.")

    # in_files = utils.get_file_paths(g_args.input, *IN_IMG_EXT)


def exit(msg=None):
    if msg:
        print(f"{msg}")
    print("Exit script.")
    sys.exit()


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description=__description__)
    ap.add_argument(
        "-i",
        "--input",
        dest="input",
        type=utils.to_path,
        help="path to input folder",
        required=True,
        # multiple (at least one) arguments gathered into list
        # nargs='+',
    )
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        type=utils.to_path,
        help="path to output folder",
        # either argument is given or current './out' is used by default
        nargs="?",
        default=utils.join_paths(utils.get_script_dir(), "./out"),
    )
    args = ap.parse_args()
    args.script_dir = utils.get_script_dir()
    args.current_dir = utils.get_current_dir()
    return args


if __name__ == "__main__":
    # parse args
    g_args = parse_args()  # can be accessed globally
    # call main function
    main()
