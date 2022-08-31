#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
__description__ = """
	Recursively scans a directory and replaces illegal file names
	(customizable).
"""
# =============================================================================
# @author   : aleibets
# @date     : 2022/08/30
# @version  : 1.0
# =============================================================================

import argparse
import utils
import sys
import numpy as np
from tqdm import tqdm
import re
from itertools import groupby

INVALID_CHARS = {"ntfs": '[<>:/\\|?*"]|[\0-\31]', "exfat": '[<>:/\\|?*"\^]|[\0-\31]'}

# default: remove illegal chars
# use parameter '-r' to set different global replace char
REPLACE_DEFAULT = ""

# for adding custom replacements:
# these will OVERRIDE any default replacements!
# WARNING: cannot contain characters that are contained in the invalid charset used
CUSTOM_REPLACE = {
    # e.g. replace '?' with '!'
    # "?": "!"
}

# renames capital extensions (e.g. .JPG -> .jpg)
# in case there are duplicates the outputs are renamed
# image.JPG, image.jpg -> image_0.jpg, image_1.jpg
RENAME_CAPITAL_EXT = True


def replace_invalid_chars(in_string, charset, default_replacement):

    result = in_string

    # custom replacements
    for c, r in CUSTOM_REPLACE.items():
        result = result.replace(c, r)

    return re.sub(charset, default_replacement, result)


def rename_path_parts(parts):
    renamed_path_parts = []
    for p in parts:
        part = replace_invalid_chars(p, g_args.invalid_chars, g_args.replace)
        renamed_path_parts.append(part)
    return renamed_path_parts


def rename_path_if_existing(path, to_rename_paths):
    fn = utils.get_file_name(path)
    fext = utils.get_file_ext(path)
    froot = utils.get_file_path(path)
    # rename already existing output files
    cnt = 0
    while path in to_rename_paths.values():
        renamed_last = f"{fn}_{cnt}{fext}"
        path = utils.join_paths(froot, renamed_last)
        cnt += 1
    return path


def rename_files(to_rename_paths, desc):
    for src_path, dst_path in tqdm(to_rename_paths.items(), desc=desc):
        tqdm.write(f"\n{src_path} ===>")
        tqdm.write(f"{dst_path}")
        if not g_args.dry_run:
            if not utils.exists(src_path):
                # in case paths were renamed before src_path needs to be adjusted
                file_or_dir = utils.get_file_name(src_path, True)
                parts = utils.split_path(src_path)[:-1]
                renamed_parts = rename_path_parts(parts)
                src_root = "/" + "/".join(renamed_parts)
                src_path = utils.join_paths(src_root, file_or_dir)
            utils.rename_file(src_path, dst_path)


def main():

    if not utils.exists_dir(g_args.input):
        exit(f"Input directory not found: {g_args.input}")

    if g_args.character_set not in INVALID_CHARS.keys():
        exit(f"Character set not found: {g_args.character_set}")

    g_args.invalid_chars = INVALID_CHARS[g_args.character_set]

    # make sure default replacement is not set to an illegal char
    test = replace_invalid_chars(g_args.replace, g_args.invalid_chars, "-")
    if test != g_args.replace:
        exit(
            f"Replacement char can not be an invalid char from charset: c {g_args.replace} -> charset {g_args.invalid_chars}"
        )

    d = utils.to_path(g_args.input, as_string=False)

    all_dirs = []
    all_files = []
    for current_file in tqdm(d.glob("**/*"), desc="read"):
        if current_file.is_file():
            all_files.append(current_file.as_posix())
        else:
            all_dirs.append(current_file.as_posix())

    # sort
    all_dirs = utils.nat_sort_list(all_dirs)
    all_files = utils.nat_sort_list(all_files)

    # calc - dirs
    to_rename_dirs = {}
    for item in tqdm(all_dirs, desc="calc - dirs"):
        path_parts = utils.split_path(item)
        renamed_path_parts = rename_path_parts(path_parts)

        # only rename if last part is different - assumptions:
        #   * parent directories are first in list
        renamed_last = renamed_path_parts[len(renamed_path_parts) - 1]

        if renamed_last != path_parts[len(path_parts) - 1]:
            entry = "/" + "/".join(renamed_path_parts)
            entry = rename_path_if_existing(entry, to_rename_dirs)
            to_rename_dirs[item] = entry

    # calc - files
    to_rename_files = {}
    for item in tqdm(all_files, desc="calc - files"):
        path_parts = utils.split_path(item)
        renamed_path_parts = rename_path_parts(path_parts)

        renamed_file = renamed_path_parts[len(renamed_path_parts) - 1]
        # pot. check files for extensions
        if RENAME_CAPITAL_EXT:
            ext = utils.get_file_ext(renamed_file)
            fn = utils.get_file_name(renamed_file)
            lowercase = ext.lower()
            if lowercase != ext:
                renamed_file = f"{fn}{lowercase}"
                renamed_path_parts[len(renamed_path_parts) - 1] = renamed_file

        # only rename if last part is different
        renamed_last = renamed_path_parts[len(renamed_path_parts) - 1]

        if renamed_last != path_parts[len(path_parts) - 1]:
            entry = "/" + "/".join(renamed_path_parts)
            entry = rename_path_if_existing(entry, to_rename_files)
            to_rename_files[item] = entry

    tqdm.write("\n")
    tqdm.write(
        "----------------------------------------- [SUMMARY] -----------------------------------------"
    )
    tqdm.write(f"               #dirs   : {len(to_rename_dirs.keys())}")
    tqdm.write(f"               #files  : {len(to_rename_files.keys())}")
    tqdm.write(f"               dry-run : {g_args.dry_run}")
    tqdm.write(
        "---------------------------------------------------------------------------------------------"
    )

    tqdm.write(
        "\n####################################### [DIRECTORIES] #######################################"
    )

    # process dirs
    rename_files(to_rename_dirs, "process - dirs")

    tqdm.write(
        "\n########################################## [FILES] ##########################################"
    )

    # process files
    rename_files(to_rename_files, "process - files")


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
        "-c",
        "--character-set",
        dest="character_set",
        type=str,
        help=f"character set: {INVALID_CHARS.keys()}",
        # either argument is given or current './out' is used by default
        nargs="?",
        default="ntfs",
    )
    ap.add_argument(
        "-r",
        "--replace",
        dest="replace",
        type=str,
        help="default replacement char",
        # either argument is given or current './out' is used by default
        nargs="?",
        default=REPLACE_DEFAULT,
    )
    ap.add_argument(
        "-d",
        "--dry-run",
        dest="dry_run",
        help="dry run - no files will be changed",
        action="store_true",
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
