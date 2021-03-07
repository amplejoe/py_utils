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
# @Date: 2020-01-27T11:53:43+01:00
# @Filename: main_template.py
# @Last modified by: aleibets
# @Last modified time: 2020-02-12T16:07:40+01:00
# @description:

import argparse
from py_utils import utils
import sys
import numpy as np
from tqdm import tqdm

IN_EXTENSIONS = [".jpg", ".png"]


def main():

    g_args.input = utils.to_path(g_args.input)
    g_args.output = utils.to_path(g_args.output)

    if g_args.input == g_args.output:
        exit("IN cannot be the same as OUT path.")

    # if not (utils.confirm_overwrite(g_args.output, "n")):
    #     exit("Aborted folder creation.")

    # in_files = utils.get_file_paths(g_args.input, *IN_EXTENSIONS)


def exit(msg=None):
    if msg:
        print(f"{msg}")
    print("Exit script.")
    sys.exit()


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to input folder",
        required=True,
        dest="input",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to output folder",
        required=True,
        dest="output",
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # parse args
    g_args = parse_args()  # can be accessed globally
    # call main function
    main()
