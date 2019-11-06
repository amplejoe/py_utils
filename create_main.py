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
# @Date: 2019-11-06T13:34:10+01:00
# @Filename: create_main.py
# @Last modified by: aleibets
# @Last modified time: 2019-11-06T13:34:17+01:00
# @description:


import argparse
import sys
import utils

TEMPLATE_FILE = "main_template.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_path")
    args = parser.parse_args()

    cwd = utils.get_current_dir()
    scd = utils.get_script_dir()

    template_path = utils.join_paths_str(scd, TEMPLATE_FILE)

    out_path = utils.to_path_str(args.target_path)
    if not utils.is_absolute_path(out_path):
        out_path = utils.join_paths_str(cwd, out_path)
    # file_name = utils.get_file_name(out_path)

    if utils.exists_dir(out_path):
        print(f"Error: target path is a directory!")
        sys.exit(0)
    elif utils.exists_file(out_path):
        utils.confirm_delete_file(out_path)

    utils.copy_to(template_path, out_path)
    print(f"Created file {out_path}")


# call main function
main()
