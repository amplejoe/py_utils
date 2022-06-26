#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
__description__ = """
	Creates symlinks to all files in a specific directory at a specified
	target root directory. Allows for keeping directory structure or not.
"""
# =============================================================================
# @author   : aleibets
# @date     : 2022/06/26
# @version  : 1.0
# =============================================================================

import argparse
from py_utils import utils
import sys
import numpy as np
from tqdm import tqdm
import os

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

    if not (utils.confirm_overwrite(g_args.output, "n")):
        print("Skip folder creation.")

    in_files = utils.get_file_paths(g_args.input)

    for f in tqdm(in_files):
        fn = utils.get_file_name(f)
        out_root = g_args.output
        if g_args.keep_dir_structure:
          rel_path = utils.path_to_relative_path(f, g_args.input, remove_file=True)
          out_root = utils.join_paths(g_args.output, rel_path)
          utils.make_dir(out_root)
        lnk = utils.join_paths(out_root, fn)
        if utils.exists_file(lnk):
          tqdm.write(f"Skipping existing link: {lnk}")
        else:
          os.symlink(f, lnk)
          tqdm.write(f"Created: {f} <===> {lnk}")


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
        help="path to input folder - contains all files one wants to link",
        required=True,
        # multiple (at least one) arguments gathered into list
        # nargs='+',
    )
    ap.add_argument(
        "-k",
        "--keep-dir-structure",
        dest="keep_dir_structure",
        help="force keeping original directory structure",
        action='store_true'
    )
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        type=utils.to_path,
        help="path to output directory - files will be linked there under their original names",
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
