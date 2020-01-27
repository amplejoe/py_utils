#!/usr/bin/env python

import argparse
from py_utils import utils
import sys
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_root")
    parser.add_argument("out_root")
    args = parser.parse_args()

    in_root = utils.to_path_str(args.in_root)
    out_root = utils.to_path_str(args.out_root)

    # in_files = utils.get_file_paths(in_root, ".jpg", ".png")


# call main function
main()
