#!/usr/bin/env python

###
# File: create_main.py
# Created: Tuesday, 12th May 2020 5:46:35 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:12:58 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###

import argparse
import sys
import utils

TEMPLATE_FILE = "main_template.py"
REPLACE_TEXT = "py_utils"

def replace_comment_filename(file):
    fn_full = utils.get_full_file_name(file)
    fin = open(file, "rt")
    data = fin.read()
    data = data.replace(TEMPLATE_FILE, fn_full)
    fin.close()
    fin = open(file, "wt")
    fin.write(data)
    fin.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_path")
    args = parser.parse_args()

    cwd = utils.get_current_dir()
    scd = utils.get_script_dir()
    # utils_dir_name = utils.get_nth_parentdir(scd)

    template_path = utils.join_paths_str(scd, TEMPLATE_FILE)

    out_path = utils.to_path(args.target_path)
    if not utils.is_absolute_path(out_path):
        out_path = utils.join_paths_str(cwd, out_path)
    out_dir = utils.get_file_path(out_path)
    # file_name = utils.get_file_name(out_path)

    if utils.exists_dir(out_path):
        print(f"Error: target path is a directory!")
        sys.exit(0)
    elif utils.exists_file(out_path):
        if not utils.confirm_delete_file(out_path):
            utils.exit("Aborted")

    # copy template
    utils.copy_to(template_path, out_path)
    print(f"Created file {out_path}")

    if out_dir not in cwd:
        # py_utils path is not relative to created file
        print(
            "Important: Please make sure that python utils are available, i.e. inside PYTHONPATH."
        )
        print("Clone repository via:")
        print("git clone https://github.com/amplejoe/py_utils.git")
    else:
        # py_utils path is relative to created file
        replace_string = REPLACE_TEXT

        # find relative path
        rel_path_to_utils = cwd.replace(f"{out_dir}", "")
        if rel_path_to_utils.startswith("/"):
            # chop first '/'
            rel_path_to_utils = rel_path_to_utils[1:]
        # replace remaining '/' with '.'
        utils_import_string = rel_path_to_utils.replace("/", ".")
        if utils_import_string == "":
            # don't import from specific folder
            replace_string = f"from {replace_string} "

        # replace utils path in copied file
        fin = open(out_path, "rt")
        data = fin.read()
        data = data.replace(replace_string, utils_import_string)
        fin.close()
        fin = open(out_path, "wt")
        fin.write(data)
        fin.close()
    replace_comment_filename(out_path)


# call main function
main()
