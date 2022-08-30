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


def main():

    if not utils.exists_dir(g_args.input):
        exit(f"Input directory not found: {g_args.input}")

    if g_args.character_set not in INVALID_CHARS.keys():
        exit(f"Character set not found: {g_args.character_set}")

    invalid_chars = INVALID_CHARS[g_args.character_set]

    # make sure default replacement is not set to an illegal char
    test = replace_invalid_chars(g_args.replace, invalid_chars, "-")
    if test != g_args.replace:
        exit(
            f"Replacement char can not be an invalid char from charset: c {g_args.replace} -> charset {invalid_chars}"
        )

    d = utils.to_path(g_args.input, as_string=False)

    all_dirs_files = []
    # DON'T use tqdm here:
    # it spams progress bars in other tools (and introducing a flag breaks backward compatibility - use get_files instead)
    for current_file in tqdm(d.glob("**/*"), desc="read"):
        all_dirs_files.append(current_file.as_posix())

    all_dirs_files_sorted = utils.nat_sort_list(all_dirs_files)

    # renaming
    to_rename = {}
    for item in tqdm(all_dirs_files_sorted, desc="rename"):
        cur_item_name = utils.get_file_name(item, True)
        cur_item_path = utils.get_nth_parentdir(item)
        renamed = replace_invalid_chars(
            cur_item_name, invalid_chars, g_args.replace
        )
        if utils.exists_file(cur_item_name):
            # pot. check files for extensions
            if RENAME_CAPITAL_EXT:
                ext = utils.get_file_ext(renamed)
                fn = utils.get_file_name(renamed)
                lowercase = ext.lower()
                if lowercase != ext:
                    renamed = f"{fn}.{lowercase}"
        if renamed != cur_item_name:
            to_rename[item] = utils.join_paths(cur_item_path, renamed)

    # replacing
    for src_path, dst_path in tqdm(to_rename.items(), desc="replacing"):
        final_dst = dst_path
        final_dst_root = utils.get_nth_parentdir(final_dst)
        cnt = 0

        if utils.exists_file(src_path):
            # file
            # rename dst if existing
            while utils.exists_file(final_dst):
                ext = utils.get_file_ext(final_dst)
                fn = utils.get_file_name(final_dst)
                final_dst_name = f"{fn}_{cnt}.{ext}"
                final_dst = utils.join_paths(final_dst_root, final_dst_name)
                cnt += 1
        else:
            # dir
            # rename dst if existing
            while utils.exists_file(final_dst):
                final_dst = f"{final_dst}_{cnt}"
                final_dst = utils.join_paths(final_dst_root, final_dst_name)
                cnt += 1

        tqdm.write(f"{src_path} ===> {final_dst}")
        if not g_args.dry_run:
            utils.rename_file(src_path, final_dst)


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
