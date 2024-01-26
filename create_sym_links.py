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

import sys
import argparse
from py_utils import utils
import sys
import numpy as np
from tqdm import tqdm
import os

MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)

IN_IMG_EXT = [".jpg", ".png"]
IN_VID_EXT = [".mp4", ".avi", ".mov"]
IN_DATA_EXT = [".npy", ".csv", ".json"]

MODES = ['target', 'files', 'folders']
DEFAULT_MODE = 'target'

def main():

    if (
        g_args.output == g_args.input
        or g_args.output == "."
        or g_args.output == g_args.script_dir
    ):
        exit("IN cannot be the same as OUT, '.' or script path.")


    if g_args.mode not in MODES:
        exit(f"Invalid mode given: {g_args.mode}")

    in_items = []

    if g_args.mode == 'target':
        # link target only, e.g. ./my_link <==> ./some/folder
        in_items.append(utils.to_path(g_args.input))
    else:
        # link all sub files or folders (optionally recursive)
        if g_args.recursive:
            if g_args.mode == 'files':
                in_items = utils.get_file_paths(g_args.input)
            else:
                in_items = utils.get_folders(g_args.input, False)
        else:
            if g_args.mode == 'files':
                in_items = utils.get_immediate_subfiles(g_args.input)
            else:
                in_items = utils.get_immediate_subdirs(g_args.input)


    for source in tqdm(in_items):
        fn_full = utils.get_file_name(source, True)
        out_root = g_args.output
        if g_args.keep_dir_structure:
            rel_path = utils.path_to_relative_path(source, g_args.input, remove_file=True)
            out_root = utils.join_paths(g_args.output, rel_path)
        # do NOT join paths here as symlinks are resolved - build own path instead
        dest = f"{utils.to_path(out_root)}/{fn_full}"
        if g_args.mode == 'target':
            dest = utils.to_path(out_root)

        # dont use utils.confirm_overwrite here - they follow symlinks
        if (utils.exists_file(dest) or utils.exists_dir(dest)):
            do_overwrite = False
            if g_args.yes:
                do_overwrite = True
            elif g_args.no:
                do_overwrite = False
            else:
               do_overwrite = utils.confirm(f"File exists: {dest} - overwrite?", "y")

            if not do_overwrite:
                tqdm.write(f"Skipping existing link: {dest}")
            else:
                try:
                    tqdm.write(f"Overwriting existing link: {dest}")
                    if utils.exists_file(dest) or utils.exists_dir(dest):
                        os.unlink(dest)
                    if g_args.mode != 'target':
                        utils.make_dir(out_root)
                    os.symlink(source, dest)
                    tqdm.write(f"Created: {source} <===> {dest}")
                except Exception as e:
                    tqdm.write(f"Error creating link: {dest}")
                    tqdm.write(
                        "  -> Windows users: this script requires enabled Developer Mode."
                    )
                    tqdm.write(str(e))


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
        help="path to input folder - contains all files/folders one wants to link",
        required=True,
        # multiple (at least one) arguments gathered into list
        # nargs='+',
    )
    ap.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        help=f"operation mode: {MODES}",
        nargs="?",
        default=DEFAULT_MODE,
    )
    ap.add_argument(
        "-k",
        "--keep-dir-structure",
        dest="keep_dir_structure",
        help="force keeping original directory structure",
        action="store_true",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        help="recursively traverse source dir",
        action="store_true",
    )
    ap.add_argument(
        "-y",
        "--yes",
        dest="yes",
        help="yes to all confirmations (CAUTION: data could be overwritten!)",
        action="store_true",
    )
    ap.add_argument(
        "-n",
        "--no",
        dest="no",
        help="no to all confirmations",
        action="store_true",
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
