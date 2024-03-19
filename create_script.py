#!/usr/bin/env python3

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
import textwrap


TEMPLATE_OPTIONS = {
    "main": "./templates/main_template.py",
    "class": "./templates/class_template.py",
    "bash": "./templates/bash_template.sh"
}
DEFAULT_TEMPLATE = "main"

PY_UTILS_DIR = "py_utils"
PY_UTILS_FILE = "utils.py"
COPY_UTILS = False

HEADER_DESCRIPTION = "[DESCRIPTION]"
HEADER_AUTHOR = "[AUTHOR]"
HEADER_DATE = "[DATE]"
CLASS_NAME = "CLASS_NAME"

def format_description(desc, width=70):
    desc = textwrap.wrap(desc, width)
    desc = "\n".join(desc)
    return textwrap.indent(desc, "\t")

def main():

    # user input
    if not g_args.type:
        g_args.type = utils.read_string_input(msg="type", init_value=DEFAULT_TEMPLATE)
    if not g_args.output:
        g_args.output = utils.read_path_input(msg="file path")
    g_args.author = utils.read_string_input(msg="author", init_value=g_args.author)
    if not g_args.description:
        g_args.description = utils.read_string_input(msg="description", init_value=HEADER_DESCRIPTION)
    g_args.description = format_description(g_args.description)
    if g_args.type != "bash":
        g_args.copy_utils = utils.confirm("Copy utilities?", "y" if g_args.copy_utils else "n")

    template_path = utils.join_paths_str(g_args.script_dir, TEMPLATE_OPTIONS[g_args.type])

    if utils.exists_dir(g_args.output):
        print("Error: target path is a directory!")
        sys.exit(0)
    elif utils.exists_file(g_args.output):
        if not utils.confirm_delete_file(g_args.output, "n"):
            utils.exit("Aborted")

    out_folder = utils.get_file_path(g_args.output)
    out_file = utils.get_file_name(g_args.output)


    if not utils.exists_dir(out_folder):
        utils.make_dir(g_args.output)

    # copy template
    utils.copy_to(template_path, g_args.output)
    print(f"Created file {g_args.output}")

    if g_args.type == "class":
        utils.replace_file_text(g_args.output, CLASS_NAME, out_file)

    if g_args.type != "bash":
        if g_args.copy_utils:
            utils_folder = PY_UTILS_DIR
            out_py_utils_dir = utils.join_paths(out_folder, utils_folder)
            utils.make_dir(out_py_utils_dir)
            utils.copy_to(PY_UTILS_FILE, out_py_utils_dir)
            print(f"Created file {out_py_utils_dir}/{PY_UTILS_FILE}")
        else:
            print(
            """
            Important: Please make sure that python utils are available, i.e. inside PYTHONPATH.
            Clone repository via: git clone https://github.com/amplejoe/py_utils.git
            """
            )

    # header information
    date = utils.get_date_str()
    utils.replace_file_text(g_args.output, HEADER_DATE, date)
    utils.replace_file_text(g_args.output, HEADER_AUTHOR, g_args.author)
    if g_args.description:
        utils.replace_file_text(g_args.output, HEADER_DESCRIPTION, g_args.description)



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        type=utils.to_path,
        help="output file path"
    )
    ap.add_argument(
        "-t",
        "--type",
        dest="type",
        type=str,
        help=f"script type ({TEMPLATE_OPTIONS.keys()})",
        choices=TEMPLATE_OPTIONS.keys()
    )
    ap.add_argument(
        "-a",
        "--author",
        dest="author",
        type=str,
        help="author name",
        nargs="?",
        default=utils.get_user_name()
    )
    ap.add_argument(
        "-d",
        "--description",
        dest="description",
        type=str,
        help="script description"
    )
    ap.add_argument(
        "-c",
        "--copy-utils",
        dest="copy_utils",
        action="store_true",  # default: False
        help="copy utils to output folder",
    )
    args = ap.parse_args()
    args.script_dir = utils.get_script_dir()
    args.current_dir = utils.get_current_dir()
    return args


if __name__ == "__main__":
    g_args = parse_args()
    main()
