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

    g_args["in"] = utils.to_path_str(g_args["in"])
    g_args["out"] = utils.to_path_str(g_args["out"])

    if g_args["in"] == g_args["out"]:
        exit("IN cannot be the same as OUT path.")

    # if not (utils.confirm_overwrite(g_args["out"], "n")):
    #     exit("Aborted folder creation.")

    # in_files = utils.get_file_paths(g_args["in"], *IN_EXTENSIONS)


def exit(msg=None):
    if msg:
        print(f"{msg}")
    print("Exit script.")
    sys.exit()


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in", type=str,
                    help="path to input folder", required=True)
    ap.add_argument("-o", "--out", type=str,
                    help="path to output folder", required=True)
    args = vars(ap.parse_args())
    return args


if __name__ == "__main__":
    # parse args
    g_args = parse_args()   # can be accessed globally
    # call main function
    main()
