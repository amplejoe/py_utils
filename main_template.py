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
        dest="input",
        type=str,
        help="path to input folder",
        required=True,
        # multiple (at least one) arguments gathered into list
        # nargs='+',
    )
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="path to output folder",
        # either argument is given or current dir is used by default
        nargs="?",
        default=".",
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # parse args
    g_args = parse_args()  # can be accessed globally
    # call main function
    main()
