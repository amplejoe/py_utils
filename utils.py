#!/usr/bin/env python

###
# File: utils.py
# Created: Tuesday, 12th May 2020 5:46:35 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: May 2022
# Modified By: Andreas (amplejoe@gmail.com)
# -----
#
# IMPORTANT: python 3.6+ required
###

#### ------------------------------------------------------------------------------------------ ####
####    IMPORTS
#### ------------------------------------------------------------------------------------------ ####


from io import TextIOWrapper
import os
import sys
import errno
import json
import simplejson
from datetime import datetime
import shutil
from natsort import natsorted
import pathlib
import re
import subprocess
import numpy as np
import inspect
import functools
import math
from tqdm import tqdm
import shlex
import decimal
import concurrent.futures
import tempfile
import getpass
from dictlib import Dict

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer
from prompt_toolkit.completion import PathCompleter
import wslconv

import textwrap
import pandas as pd
import csv
import typing
import timeit


#### ------------------------------------------------------------------------------------------ ####
####    USER INPUT
#### ------------------------------------------------------------------------------------------ ####


def select_option(options, *, msg=None, default=0):
    """
    Ask user to select one of several options.
    Parameters
    ----------
    options: list of strings
        options to select
    msg: string
        user prompt message
    default: integer
        default selected idx
    return: tuple (integer, object)
        idx and value of selected option

    """
    if default < 0 or default > len(options) - 1:
        default = None
    accepted_answers = list(range(0, len(options)))
    user_prompt = f"[0-{len(options)-1}]"
    if default is not None and default in accepted_answers:
        accepted_answers.append("")  # allows users to press enter
        user_prompt = user_prompt + f" ({default})"
    if msg is not None:
        print(msg)
    answer = -1

    for i, o in enumerate(options):
        if default == i:
            print(f"{i}. [{o}]")
        else:
            print(f"{i}. {o}")

    while answer not in accepted_answers:
        answer = input(f"user_prompt ")
        if answer == "":
            answer = default
        answer = int(answer)
    return answer, options[answer]


def confirm(msg=None, default=None):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    if default != None:
        default = default.lower()
    accepted_answers = ["y", "n"]
    user_prompt = "[y/n]"
    if default is not None and default in accepted_answers:
        default = default.lower()
        accepted_answers.append("")  # allows users to press enter
        user_prompt = user_prompt.replace(default, default.upper())
    if msg is not None:
        print(msg)
    answer = None
    while answer not in accepted_answers:
        answer = input(user_prompt).lower()
        if answer == "":
            answer = default
    return answer == "y"


def confirm_delete_file(path, default=None, msg=None):
    p = to_path(path, as_string=False)
    out_msg = "File exists: %s, delete file?" % (path)
    if msg:
        out_msg = msg
    if p.is_file():
        if confirm(out_msg, default):
            remove_file(path)
        else:
            # user explicitly typed 'n'
            return False
    return True


def confirm_delete_path(path, default=None, msg=None):
    p = to_path(path, as_string=False)
    out_msg = "File exists: %s, delete file?" % (path)
    if msg:
        out_msg = msg
    if p.is_dir():
        if confirm(out_msg, default):
            remove_dir(path)
        else:
            # user explicitly typed 'n'
            return False
    return True


def confirm_overwrite(path, default=None, msg=None):
    """Confirms overwriting a path or a file.
    Dermines items to create by file extension, hence,
    will NOT create new directories if path contains a '.'.
    """
    p = to_path(path, as_string=False)
    confirmed = False
    if p.is_dir():
        confirmed = confirm_delete_path(path, default)
        make_dir(path, True)
    elif p.is_file():
        confirmed = confirm_delete_file(path, default)
    else:
        confirmed = True
        # TODO:  improve this or create separate function for files
        # here we don't know if the new element should be a file or a directory
        # -> decide by checking for extension
        ext = get_file_ext(path)
        if ext == "":
            make_dir(path, True)
    return confirmed


def complete_path(text, state):
    """Path autocompletion like bash."""
    incomplete_path = pathlib.Path(text)
    if incomplete_path.is_dir():
        completions = [p.as_posix() for p in incomplete_path.iterdir()]
    elif incomplete_path.exists():
        completions = [incomplete_path]
    else:
        exists_parts = pathlib.Path(".")
        for part in incomplete_path.parts:
            test_next_part = exists_parts / part
            if test_next_part.exists():
                exists_parts = test_next_part

        completions = []
        for p in exists_parts.iterdir():
            p_str = p.as_posix()
            if p_str.startswith(text):
                completions.append(p_str)
    return completions[state]


def read_string_input(*, init_value="", msg="input text"):
    """Reads a string from user input.

    Args:
        init_value (str, optional): [description]. Defaults to "".
        msg (str, optional): [description]. Defaults to "input text".

    Returns:
        [type]: [description]
    """
    result = prompt(f"{msg}: ", default=init_value)
    return result


# OLD: problems with different keyboard layouts
# TODO: test on windows, then remove
# def read_string_input(*, init_value="", msg="input text"):
#     """Reads a string from user input.

#     Args:
#         init_value (str, optional): [description]. Defaults to "".
#         msg (str, optional): [description]. Defaults to "input text".

#     Returns:
#         [type]: [description]
#     """
#     # only works on linux - use pyautogui instead
#     # readline.set_startup_hook(lambda: readline.insert_text(init_value))
#     print("----------------")
#     print(init_value)
#     print("----------------")
#     typewrite(init_value)
#     return input(f"{msg}: ")


def read_path_input(*, init_path=None, msg="input path"):
    """Reads a path from user input. Supports path completion.

    Args:
        init_path ([type], optional): [description]. Defaults to None.
        msg (str, optional): [description]. Defaults to "input path".

    Returns:
        [type]: [description]
    """
    if init_path is None:
        init_path = get_user_home_dir()

    init_path = str(to_path(init_path, as_string=False)) # completer can only handle system specific paths, so omit posix conversion

    completer = PathCompleter()
    result = prompt(f"{msg}: ", default=init_path, completer=completer)
    return to_path(result)


#### ------------------------------------------------------------------------------------------ ####
####    FILE OPERATIONS
#### ------------------------------------------------------------------------------------------ ####


def exists_dir(*p):
    """Checks whether a directory really exists."""
    return to_path(*p, as_string=False).is_dir()


def exists_file(*p):
    """Checks whether a file really exists."""
    return to_path(*p, as_string=False).is_file()


def exists(*p):
    """Checks whether a path (file or dir) really exists."""
    return to_path(*p, as_string=False).exists()


def is_dir_path(path):
    """Rudimentary check if (non-existing) path is a directory.
        WARNING: only checks for '.' in last path part (files without extension are ignored!)

    Args:
        path (str): input path
    """
    fn = get_full_file_name(path)

    if "." in fn:
        return False
    else:
        return True


def is_file_path(path):
    """Rudimentary check if (non-existing) path contains a file.
        WARNING: only checks for '.' in last path part (files without extension are ignored!)

    Args:
        path (str): input path
    """
    fn = get_full_file_name(path)

    if "." in fn:
        return True
    else:
        return False


def to_path_url(*p, as_string=True):
    """Convert URL to pathlib path.
    INFO: Use this with URLS, as to_path will raise an error with URLS
    """
    pth = pathlib.Path(*p)
    if as_string:
        return pth.as_posix()
    else:
        return pth


def to_path_url_str(*p):
    """Convert URL string to pathlib path.
    Returns string representation.
    ------
    deprecated - to_path_url already per default returns string
    """
    return to_path_url(*p)


def to_path(*p, as_string=True):
    """Convert string to pathlib path.
    INFO: Path resolving removes stuff like ".." with 'strict=False' the
    path is resolved as far as possible -- any remainder is appended
    without checking whether it really exists.
    """
    pl_path = pathlib.Path(*p)
    ret = pl_path.resolve(strict=False)  # default return in case it is absolute path

    if not pl_path.is_absolute():
        # don't resolve relative paths (pathlib makes them absolute otherwise)
        ret = pl_path

    if as_string:
        return ret.as_posix()
    else:
        return ret


def to_path_str(*p):
    """Convert string to pathlib path.
    Returns string representation.
    --------
    deprecated - 'to_path' does the same with as_string=True
    """
    return to_path(*p)


def split_path(p, include_dir=False):
    """Converts path string to list with one entry per path part.
    WARNING: will convert any relative paths to absolute paths (corresponding to executing script).
    ----------------------
    include_dir: include root directory
    returns: list(str) - the first entry is either the directory root or not set as defined by include_dir
    """
    pl_path = to_path(p, as_string=False)
    pl_path_res = pl_path.resolve()

    p_list = pathlib.PurePosixPath(pl_path_res).parts

    if include_dir:
        return p_list
    else:
        return p_list[1:]


def join_paths(path, *paths, as_string=True):
    """Joins path with arbitrary amount of other paths."""
    joined = to_path(path, as_string=False).joinpath(to_path(*paths, as_string=False))
    joined_resolved = to_path(joined, as_string=False)
    if as_string:
        return joined_resolved.as_posix()
    else:
        return joined_resolved


def join_paths_str(path, *paths):
    """Joins path with arbitrary amount of other paths.
    Returns string representation.
    --------
    deprecated - 'join_paths' does the same with as_string=True
    """
    return join_paths(path, *paths)


def make_temp_dir(path, prefix="", show_info=False):
    """Creates a temp directory
    Parameters
    ----------
    path:
        the directory path
    show_info:
        show creation user infos (default=False)
    """
    try:
        if not exists_dir(path):
            make_dir(path)
        tempdir = tempfile.mkdtemp(dir=path, prefix=prefix)
    except OSError as e:
        tqdm.write("Unexpected error: %s", str(e.errno))
        raise  # This was not a "directory exists" error..
    if show_info:
        tqdm.write(f"Created dir: {path}")
    return tempdir


def make_dir(path, show_info=False, overwrite=False):
    """Creates a directory
    Parameters
    ----------
    path:
        the directory path
    overwrite:
        force directory overwrite (default=False)
    show_info:
        show creation user infos (default=False)
    """
    try:
        if overwrite:
            remove_dir(path)
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            tqdm.write("Unexpected error: %s", str(e.errno))
            raise  # This was not a "directory exists" error..
        # tqdm.write("Directory exists: %s", path)
        return False
    if show_info:
        tqdm.write(f"Created dir: {path}")
    return True


def get_owner(path):
    """Gets owner and group of a path.

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    path = to_path(path, as_string=False)
    if exists_dir(path):
        return path.owner(), path.group()
    return False, False


def change_owner(path, user, group, silent=False):
    """Changes the user and group of a path. IMPORTANT: requires root privileges in Linux.

    Args:
        path (_type_): _description_
        user (_type_): _description_
        group (_type_): _description_
    """
    if exists_dir(path):

        # alternative
        # import pwd
        # import grp
        # uid = pwd.getpwnam(user).pw_uid
        # gid = grp.getgrnam(group).gr_gid
        # os.chown(path, uid, gid)

        try:
            shutil.chown(path, user, group)
            if not silent:
                print(f"Changed owner for {path} -> {user}:{group}")
            return True
        except OSError as e:
            raise e


def remove_dir(path):
    if exists_dir(path):
        # os.rmdir(path) # does not work if not empty
        shutil.rmtree(path, ignore_errors=True)  # ignore errors on windows


def clear_directory(path):
    """Clears content of a directory without deleting the directory itself.

    Args:
        path ([type]): [description]
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            tqdm.write("Failed to delete %s. Reason: %s" % (file_path, e))


def get_directories(directory):
    retList = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            # print os.path.join(root, filename)
            retList.append(os.path.abspath(os.path.join(root, d)))
    return retList


def get_immediate_subdirs(a_dir, full_path=True):
    """Returns a list of immediate sub-directories of a path.
    full_path (Default: True): get full path or sub-dir names only
    """
    path = to_path(a_dir, as_string=False)
    if not path.is_dir():
        return []
    if full_path:
        return [f.as_posix() for f in path.iterdir() if f.is_dir()]
    else:
        return [get_nth_parentdir(f) for f in path.iterdir() if f.is_dir()]


def get_immediate_subfiles(a_dir, full_path=True):
    """Returns a list of immediate sub-files of a path.
    full_path (Default: True): get full path or sub-file names only
    """
    path = to_path(a_dir, as_string=False)
    if not path.is_dir():
        return []
    if full_path:
        return [f.as_posix() for f in path.iterdir() if f.is_file()]
    else:
        return [get_nth_parentdir(f) for f in path.iterdir() if f.is_file()]


def get_nth_parentdir(file_path, n=0, full_path=False):
    """Get nth parent directory of a file path, starting from the back (file).
    (Default: 0, i.e. the first directory after a potential filename)
    full_path: return full path until nth parent dir
    """
    p = to_path(file_path, as_string=False)
    ret_path = None
    if p.is_dir():
        n = n - 1
    if (n < 0) or (n > (len(p.parents) - 1)):
        ret_path = p
    else:
        ret_path = p.parents[n]
    if not full_path:
        return ret_path.name
    else:
        return ret_path.as_posix()


def is_dir_empty(path):
    """
    Check if a Directory is empty or is non existent
    """
    if len(os.listdir(path)) == 0 or not os.path.isdir(path):
        return True
    else:
        return False


def get_file_path(file_path):
    """file path only (strips file from its path)"""
    p = to_path(file_path, as_string=False)
    return p.parents[0].as_posix()


def get_folders(directory, show_progress=True, *names):
    """Get all subfolders of a directory that optionally contain either value of 'names'.

    Args:
        directory (str): root folder
        show_progress (bool, optional): progress bar. Defaults to True.
        *names (str): one or more alternative strings that the last part of the returned paths should include.

    Raises:
        StopIteration: [description]

    Returns:
        [list of str]: folder paths including given names or all paths

    Yields:
        [type]: [description]
    """
    d = to_path(directory, as_string=False)

    all_files = []
    for current_path in tqdm(
        d.glob("**/*"), desc="reading folders", disable=(not show_progress)
    ):
        if current_path.is_file():
            continue
        last_part = get_nth_parentdir(current_path)

        if names:
            for n in names:
                if n in last_part:
                    all_files.append(current_path.as_posix())
                    break
        else:
            all_files.append(current_path.as_posix())

    return all_files


# supersedes get_file_paths
def get_files(directory, show_progress=True, *extensions):
    """Superseeds get_file_paths - includes toggleable progess"""
    d = to_path(directory, as_string=False)

    all_files = []
    for current_file in tqdm(
        d.glob("**/*"), desc="reading files", disable=(not show_progress)
    ):
        if not current_file.is_file():
            continue
        fext = current_file.suffix
        if extensions and fext not in extensions:
            continue
        # as_posix: Return a string representation
        # of the path with forward slashes (/)
        all_files.append(current_file.as_posix())
    return all_files


def get_file_paths(directory, *extensions):
    """Get all file paths of a directory (optionally with file extensions
    usage example: get_file_paths("/mnt/mydir", ".json", ".jpg")
    changes:
        - 2019: using pathlib (python3) now
    """
    d = to_path(directory, as_string=False)

    all_files = []
    # DON'T use tqdm here:
    # it spams progress bars in other tools (and introducing a flag breaks backward compatibility - use get_files instead)
    for current_file in d.glob("**/*"):
        if not current_file.is_file():
            continue
        fext = current_file.suffix
        if extensions and fext not in extensions:
            continue
        # as_posix: Return a string representation
        # of the path with forward slashes (/)
        all_files.append(current_file.as_posix())
    return all_files


def get_file_paths_containing(directory, *contained):
    """Get all file paths of all subdirectories (optionally set (partially) contained string
    usage example: get_file_paths("/mnt/mydir", "label.txt", "pic")
    """
    d = to_path(directory, as_string=False)

    all_files = []
    for current_file in d.glob("**/*"):
        if not current_file.is_file():
            continue
        # continue if none of the given contained strings match the path
        if contained and not any(x in current_file.as_posix() for x in contained):
            continue
        # as_posix: Return a string representation
        # of the path with forward slashes (/)
        all_files.append(current_file.as_posix())
    return all_files


def path_to_relative_path(path, relative_to_path, remove_file=False):
    """Return sub-path relative to input path. Optionally remove (potential) files from path end."""
    path = to_path(path, as_string=False)
    rel_to = to_path(relative_to_path, as_string=False)
    rel_pth = path.relative_to(rel_to).as_posix()
    if remove_file:
        if is_file_path(path):
            rel_pth = get_file_path(rel_pth)
    return rel_pth


def get_full_file_name(file_path):
    """full file name plus extension"""
    return to_path(file_path, as_string=False).name


def get_file_name(file_path, full=False):
    """Get file name of file"""
    if full:
        return to_path(file_path, as_string=False).name
    else:
        return to_path(file_path, as_string=False).stem


def get_file_ext(file_path):
    """Get file extension of file (always with '.', i.e. 'test.jpg' -> '.jpg')"""
    return to_path(file_path, as_string=False).suffix


def add_suffix_to_file(file_path, suffix):
    """Adds a suffix to file path. No additional character is added, i.e. '_' needs to be part of suffix if required.

    Args:
        file_path ([type]): [description]
        suffix ([type]): The suffix. E.g. "_suffix"

    Returns:
        [type]: [description]
    """
    fp = get_file_path(file_path)
    fn = get_file_name(file_path)
    fe = get_file_ext(file_path)
    return join_paths(fp, f"{fn}{suffix}{fe}")


def add_suffix_to_path(pth, suffix):
    """Adds a suffix to a path. No additional character is added, i.e. '_' needs to be part of suffix if required.

    Args:
        pth ([type]): [description]
        suffix ([type]): The suffix. E.g. "_suffix"

    Returns:
        [type]: [description]
    """
    res = to_path(pth)
    return f"{res}{suffix}"


def remove_file(path):
    """Removes file (unlink) if existing.

    Args:
        path (string): file path
    """
    p = to_path(path, as_string=False)
    if exists_file(p):
        try:
            p.unlink()
        except OSError as e:
            print("Error: %s : %s" % (p, e.strerror))


def copy_to(src_path, dst_path, follow_symlinks=True, ignore_list=None):
    """Copies src_path to dst_path.
    If dst is a directory, a file with the same basename as src is created
    (or overwritten) in the directory specified.
    ignore_pattern:
        shutil ignore pattern (see doc, e.g. ignore_list = ['*.pyc', 'tmp*'])
    """
    try:
        if os.path.isdir(src_path):
            # copy dir recursively
            if ignore_list and len(ignore_list) > 0:
                shutil.copytree(
                    src_path,
                    dst_path,
                    symlinks=follow_symlinks,
                    ignore=shutil.ignore_patterns(*ignore_list),
                )
            else:
                shutil.copytree(src_path, dst_path, symlinks=follow_symlinks)
        else:
            # copy file
            shutil.copy(src_path, dst_path, follow_symlinks=follow_symlinks)
        return True
    except IOError as e:
        tqdm.write(f"Unable to copy file. {e}")
        return False


def move_file(src_path, dst_path):
    """Moves a file using shutil. Can also be used for file renaming

    Args:
        src (string): path to src name
        dst (string): path to dst name
    """
    return shutil.move(src_path, dst_path)


def rename_file(src_path, dst_path):
    """Uses move_file to rename file"""
    src_root = to_path(get_file_path(src_path))
    dst_root = to_path(get_file_path(dst_path))
    if src_root != dst_root:
        tqdm.write(f"Cannot rename file: {src_root}")
        tqdm.write("Paths don't match!")
        exit(0)
    return shutil.move(src_path, dst_path)


def is_absolute_path(path):
    """Checks if path is absolute (also for non-existing paths!)."""
    if path is None:
        return False
    return os.path.isabs(path)


def rel_to_abs_path(path, rel_to_path=None):
    """Converts relative path to absolute path depending on current or custom (rel_to_path) dir"""
    if is_absolute_path(path):
        return path
    target_dir = get_current_dir()
    if rel_to_path is not None:
        target_dir = rel_to_path
    return join_paths_str(target_dir, path)


def read_json(path, silent=True):
    """Loads a json file into a variable.
    Parameters:
    ----------
    path : str
        path to json file
    Return : dict
        dict containing all json fields and attributes
    """
    data = None
    try:
        with open(path) as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        if not silent:
            print(f"File not found: {path}")
    return data


def print_dict_pretty(in_dict, indent_amount=4):
    print(json.dumps(in_dict, indent=indent_amount))


def write_json(path, data, pretty_print=False, handle_nan=False):
    """Writes a json dict variable to a file.
    Parameters:
    ----------
    path : str
        path to json output file
    data : dict
        data compliant with json format
    """
    with open(path, "w", newline="") as outfile:
        if pretty_print:
            # OLD
            # json.dump(data, outfile, indent=4)
            # json.dump(data, outfile, indent=4, sort_keys=True)
            simplejson.dump(data, outfile, indent=4, ignore_nan=handle_nan)
        else:
            # OLD
            # json.dump(data, outfile)
            simplejson.dump(data, outfile, ignore_nan=handle_nan)


def read_json_arr(json_path):
    """Load json as array of lines"""
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def get_attribute_from_json(path, attr):
    """gets attribute from json file"""
    return read_json(path)[attr]


def get_csv_dict_writer(path, headers) -> typing.Tuple[TextIOWrapper, csv.DictWriter]:
    """Gets csv writer and creates OR opens an existing csv in append mode.
       Writer usage:
            writer.writerow(dict)

       IMPORTANT: don't forget to close file when done, i.e. file.close()

    Args:
        path (str): path to file
        headers (list[str]): optional header line - ignored in append mode
    """
    fileHandle = None
    is_new_file = False
    if exists_file(path):
        fileHandle = open(path, mode="a", newline="")
    else:
        fileHandle = open(path, mode="w", newline="")
        is_new_file = True

    writer = csv.DictWriter(fileHandle, fieldnames=headers)
    if is_new_file:
        writer.writeheader()

    return fileHandle, writer


def get_csv_headers(path):
    headers = None
    with open(path, "r") as csvfile:
        datareader = csv.reader(csvfile)
        headers = next(datareader)
    return headers


def csv_row_reader(path, has_headers=True):
    """Gets csv line-by-line reader generator. Recommended for large files - for smaller files use read_csv*.
       Usage:
            Simply loop over returned generator:
            for row in get_csv_dict_reader(path):
                print(row['some_header_field'])
                # or for no headers:
                print(row[0])

       No Need to close the file after reading.

    Args:
        path (str): path to file
        has_headers (bool): indicate if the file contains a header line
    """

    with open(path, "r") as csvfile:
        datareader = csv.reader(csvfile)
        if has_headers:
            headers = next(datareader)

        for row in datareader:
            ret = None
            if has_headers:
                ret = {}
                for i, h in enumerate(headers):
                    # return dict with headers as keys
                    ret[h] = row[i]
            else:
                # return simple list
                ret = row
            yield ret


def read_csv_df(path, *, header_line: int = 0):
    """Reads csv to pandas df. Use header line=None to disable header."""
    df = pd.read_csv(path, header=header_line)
    return df


def read_csv(path, *, header_line: int = 0):
    """Reads csv to dict. Use header line=None to disable header."""
    df = read_csv_df(path, header_line=header_line)
    return df.to_dict()


def read_csv_arr(path, *, header_line=0):
    """Reads csv to list. Use header line=None to disable header."""
    df = read_csv_df(path, header_line=header_line)
    return df.values.tolist()


def get_n_last_csv_rows(path, n=1, *, header_line: int = 0):
    """Gets n rows from the end of a csv file counted from the back as a list,
    i.e. last (1), last + second to last (2), ... Default: 1 (last)
    src: https://linuxtut.com/en/02d5c656cc2faa7e35ad/
    Use header line=None to disable header.
    """
    df = read_csv_df(path, header_line=header_line)
    return df.tail(n).values.tolist()


def read_file_to_array(path, remove_new_lines=False, remove_blanks=False):
    """Reads all lines of a file into an array."""
    arr = None
    with open(path, 'r', encoding='utf-8') as file:
        if remove_new_lines:
            arr = file.read().splitlines()
        else:
            arr = file.readlines()

    if arr is not None:
        arr = [x for x in arr if (x != '' and x != '\n')]

    return arr


def write_string_to_file(str_to_write, file_path, show_info=True):
    with open(file_path, "w", newline="") as file:
        file.write(str_to_write)
    if show_info:
        tqdm.write(f"Wrote: {file_path}")


def replace_file_text(file, text, replace_text):
    """Replaces a string in a file with another string.

    Args:
        file (str): file path
        text (str): original text
        text (str): replace text
    """
    with open(file, "r+") as f:
        data = f.read()
        data = data.replace(text, replace_text)
        f.seek(0)
        f.write(data)
        f.truncate()


def find_similar_folder(target_path, folder_list):
    """Finds similar subfolder name in a directory from a list of names.
    Parameters
    ----------
    target_path : string
        path that should be searched
    folder_list : list
        list of potential folder names (will also partially match)
    """
    all_subdirs = get_immediate_subdirs(target_path, False)
    for sdir in all_subdirs:
        for f in folder_list:
            if (sdir in f) or (f in sdir):
                return sdir
    return None


def prompt_folder_confirm(target_path, folder_list, name):
    """Prompts user to confirm or enter a folder name.
    Parameters
    ----------

    target_path: string
        target root directory where folder should be searched
    folder_list: list of strings
        list of potential (partial) matches for the searched folder
    name: string
        display name as user information (e.g. 'images')
    """
    all_subdirs = get_immediate_subdirs(target_path, False)
    print(f"Finding '{name}' folder:")
    print(f"Found {len(all_subdirs)} directories: {all_subdirs}")
    sim_folder = find_similar_folder(target_path, folder_list)
    if sim_folder is not None:
        print(f"Suggested dir: '{sim_folder}' ({name})")
        if not confirm("Is this correct?", "y"):
            sim_folder = None
    if sim_folder is None:
        while (
            not (is_absolute_path(sim_folder) and exists_dir(sim_folder))
            and sim_folder not in all_subdirs
        ):
            sim_folder = input(
                f"Please enter {name} dir from list (or enter absolute path): "
            )
            if (
                not (is_absolute_path(sim_folder) and exists_dir(sim_folder))
                and sim_folder not in all_subdirs
            ):
                print("Dir not found, please try again (Ctrl+C to quit).")
    if not is_absolute_path(sim_folder):
        sim_folder = join_paths_str(target_path, sim_folder)
    print(f"Using {sim_folder} ({name}).")
    return sim_folder


#### ------------------------------------------------------------------------------------------ ####
####    DEBUG AND CONTROL
#### ------------------------------------------------------------------------------------------ ####


# from: https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def exit(msg=None):
    """Exits script with optional message."""
    if msg:
        print(f"{msg}")
    print("Exit script.")
    sys.exit()


def get_date_str():
    """Get today's date as string in form of YYYY/MM/DD"""
    today = get_date()
    return today.strftime("%Y/%m/%d")


def get_default_time():
    """Returns timeit default timer (float in seconds), i.e. start_time for show_processing_time."""
    return timeit.default_timer()


def get_processing_time(start_time, format=True):
    """Given a timeit start_time this returns the current (formatted) execution time in seconds."""
    end_processing = timeit.default_timer()
    processing_time = end_processing - start_time
    if format:
        return float_to_string(processing_time)
    else:
        return processing_time


def show_processing_time(start_time, item_name="finished"):
    """Given a timeit start_time this prints the current execution time in seconds."""
    pt_formatted = get_processing_time(start_time)
    tqdm.write(f"{item_name} - processing time: {pt_formatted} s")


def get_current_dir():
    """Returns current working directory."""
    return to_path(pathlib.Path.cwd())


def get_script_dir(resolve_symlinks=True):
    """Returns directory of currently running script (i.e. calling file of this method)."""
    # starting from 0, every file has their own frame, take the second to last's file name (= calling file frame)
    calling_file = inspect.stack()[1][1]  # [frame idx] [file name]
    # return directory of calling file
    if (resolve_symlinks):
        return os.path.dirname(os.path.abspath(calling_file))
    else:
        return os.path.dirname(calling_file)


def get_user_name():
    return getpass.getuser()


def get_user_home_dir():
    return to_path(pathlib.Path.home())


def set_environment_variable(key, value):
    """Sets an OS environment variable."""
    os.environ[key] = value


#### ------------------------------------------------------------------------------------------ ####
####    STRING MANIPULATION
#### ------------------------------------------------------------------------------------------ ####


def unindent_multiline_string(ml_string):
    """Unindents multiline strings, e.g. the following containing 3 indentations unindented:
    '''
        Hello there!
    '''
    """
    result = textwrap.dedent(ml_string)
    # make sure not to start with a blank line
    if result.startswith("\n"):
        result = result.replace("\n", "", 1)
    return result


def indent_multiline_string(ml_string):
    """Indents multiline strings, e.g. for the following each line's indentation is matched:
    '''
        print(indent_multiline_string("     first(NL)second(NL)third"))
        ->
             first
             second
             third

    '''
    """

    leading_spaces = len(ml_string) - len(ml_string.lstrip(" "))
    ml_string = ml_string.lstrip(" ")
    parts = ml_string.split("\n")

    parts = [" " * leading_spaces + x for x in parts]

    return "\n".join(parts)


def wrap_key_value_string(key: str, value: str, width: int = 80) -> str:
    """
    Generates text wrapped output of key value pair, e.g.:
        print(utils.wrap_key_value_string("seq", "a b c d e f g", width=10))
    creates:
        seq: a b c
             d e f
             g
    """
    key_str = f"{key}: "
    wrapper = textwrap.TextWrapper(
        initial_indent=key_str, width=width, subsequent_indent=" " * len(key_str)
    )
    return wrapper.fill(value)


def table_output(lst_col_sizes, lst_col_content):
    """
    Pretty tabular output where items are put out in columns with custom maximum sizes.
        Example:
        IN:     table_output([4, 10, 60], ['', 'stat:', 1234])
        OUT:    '    stat:     1234'

    """
    out_str = ""
    assert len(lst_col_sizes) == len(lst_col_content)
    # convert to strings
    lst_col_content = [str(x) for x in lst_col_content]
    for i, s in enumerate(lst_col_sizes):
        out_str += f"{lst_col_content[i]:<{s}}"
    tqdm.write(out_str)


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def set_object_attr(obj, attr, val):
    """Allows for setting object attributes recursively via strings.
       E.g: set_object_attr(person, "car.tire.brand", "Goodyear")
    Args:
        obj (object): the object to process
        attr (object): the attribute to alter
        val (object): the new vale

    Returns:
        object: the new value via get_object_attr
    """
    pre, _, post = attr.rpartition(".")
    return setattr(get_object_attr(obj, pre) if pre else obj, post, val)


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def get_object_attr(obj, attr, *args):
    """Allows for getting object attributes recursively via strings.
       E.g: get_object_attr(person, "car.tire.brand") -> "Goodyear"
    Args:
        obj (object): the object to process
        attr (object): the attribute to get
        args (list or agrs): getattr args

    Returns:
        object: the value of attr
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def left_pad_zeros(var, num_zeros):
    var = str(var)
    return var.zfill(num_zeros)


def get_regex_match_list(in_string, regex):
    matches = re.finditer(regex, in_string)
    match_list = []
    for matchNum, match in enumerate(matches, start=1):
        match_list.append(match.group().strip("'").strip('"'))
    return match_list


def get_string_before_char(input_string, stop_char):
    """Cuts out substring from input_string until a certain character.
    Info: Returns tuple: (cut out string w/o stop_char, rest of input_string w/o stop_character)
          i.e. stop_char is lost in any case!
    """
    cur_substring = ""
    rest_string = input_string
    for c in input_string:
        rest_string = rest_string[1:]  # cut 1 char from string (in any case)
        if c != stop_char:
            cur_substring += c
        else:
            break
    return cur_substring, rest_string


def get_attribute_from(file_name, attribute):
    """gets an attribute from a file/dir name as string. Converts file names / last dir from paths.
    format file: [OPTIONAL: ANY_PATH/]a1_VAL1_a2_(VAL2.txt)_a3_VAL3.EXT
    format path: [OPTIONAL: ANY_PATH/]a1_VAL1_a2_(VAL2.txt)_a3_VAL3
    Info:
      * Exceptional characters for VALs: '_', '.', '(', ')'
      * If VALs should contain '_' or '.' they MUST be encapsulated by parentheses
        (e.g. v_(my_video.mp4) -> search for "v" -> "my_video.mp4")
      * Exceptional string char: NO parentheses are allowed in VAL strings!!
      * '.EXT' is optional
    """
    file_name = get_full_file_name(file_name)  # make sure input is no path

    # parse file for attributes from beginning to end
    attribute_dict = {}

    while len(file_name) > 0:
        cur_attr, file_name = get_string_before_char(file_name, "_")
        attribute_dict[cur_attr] = None
        if len(file_name) == 0:
            break
        cur_val = ""
        if file_name[0] == "(":
            file_name = file_name[1:]  # cut '(' from string
            cur_val, file_name = get_string_before_char(file_name, ")")
            file_name = file_name[1:]  # cut '_' from string
        else:
            cur_val, file_name = get_string_before_char(file_name, "_")
            cur_val = cur_val.split(".")[0]  # strip potential file EXT
        attribute_dict[cur_attr] = cur_val

    ret_val = attribute_dict[attribute] if attribute in attribute_dict.keys() else None

    ## OLD methodology using 'split
    # attribute_split = file_name.split(f"{attribute}_")
    # ret_value = None
    # if len(attribute_split) > 1:
    #     # attribute has been found in string
    #     split_one = attribute_split[1]
    #     ret_value = split_one.split("_")[0] # default: no parentheses in VAL
    #     if "(" in ret_value:
    #         # value begins with parenthesis '('
    #         ret_value = split_one.split(")")[0].replace("(", "")
    #     else:
    #         ret_value = ret_value.split(".")[0] # strip potential file EXT

    return ret_val


def remove_invalid_file_chars(input_string):
    """
    Removes invalid chars (Windows) from string.
    """
    invalid = '<>:"/\\|?* '
    for char in invalid:
        input_string = input_string.replace(char, "")
    return input_string


def replace_right(source, target, replacement, replacements=1):
    """Replaces text in a string starting from the end.

    Args:
        source ([type]): [description]
        target ([type]): [description]
        replacement ([type]): [description]
        replacements (int, optional): how many strings to replace. Defaults to 1.

    Returns:
        [type]: [description]
    """
    return replacement.join(source.rsplit(target, replacements))


#### ------------------------------------------------------------------------------------------ ####
####    MATH AND NUMBERS
#### ------------------------------------------------------------------------------------------ ####


def float_to_string(float_var, precision=3):
    if not is_number(float_var):
        return float("NAN")
    return "%.*f" % (precision, float_var)


def get_decimals(float_number):
    """Gets the number of decimals in a float number"""
    d = decimal.Decimal(str(float_number))
    return abs(d.as_tuple().exponent)


def round_half_up(num):
    """
    Python rounds half numbers, like 0.5 down, i.e. round(0.5) = 0.
    This function rounds half numbers up, i.e. round(0.5) = 1.
    """
    res = None
    if (float(num) % 1) >= 0.5:
        res = math.ceil(num)
    else:
        res = round(num)
    return res


def safe_div(x, y):
    """Zero safe division"""
    if y == 0:
        return 0
    return x / y


def avg_list(lst):
    """Average value of a list of values"""
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def is_number(var, strict=False):
    """Checks if a variable is a number. Use strict to disallow strings containing numbers.

    Args:
        s ([type]): [description]
        strict (bool, optional): [description]. Defaults to False.

    Returns:
        bool: is a number?
    """
    try:
        # test string
        if strict and isinstance(var, str):
            return False
        # test None
        if var == None:
            return False
        # test number
        float(var)
        return True
    except ValueError:
        return False


def format_number(number, precision=3, width=0):
    """Formats a number with precision and width.

    Args:
        number (number): Any number
        precision (int, optional): decimal precision. Defaults to 3.
        width (int, optional): width of resulting string. Defaults to 0.

    Returns:
        string: formatted number as string
    """
    return f"{number:{width}.{precision}f}"


#### ------------------------------------------------------------------------------------------ ####
####    LISTS: SEARCH & MANIPULATION
#### ------------------------------------------------------------------------------------------ ####


def is_array(var):
    return isinstance(var, (list, tuple, np.ndarray))


def is_np_arr_in_list(np_array, lst):
    """Checks if a list of numpy arrays contains a specific numpy array (Identity)."""
    return next((True for elem in lst if elem is np_array), False)


def remove_np_from_list(lst, np_array):
    """Removes a numpy array from a list."""
    ind = 0
    size = len(lst)
    while ind != size and not np.array_equal(lst[ind], np_array):
        ind += 1
    if ind != size:
        lst.pop(ind)
    else:
        pass
        # raise ValueError('array not found in list.')


def is_partial_word_in_list(word, lst):
    res = filter_list_by_partial_word(word, lst)
    return len(res) > 0


def filter_list_by_partial_word(word, list_to_filter):
    """partially matches a word with a list and returns filtered list"""
    return list(filter(lambda x: word in x, list_to_filter))


def nat_sort_list(l, descending=False):
    """sorts a list naturally. Does not change original - returns sorted list!"""
    return natsorted(l, key=lambda y: y.lower(), reverse=descending)


def sort_list_nicely(l):
    """Sort the given list in the way that humans expect. Change original list (no return value)."""

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    l.sort(key=alphanum_key)


def nat_sort_dict_list_by_key(dict_list, key, descending=False):
    return natsorted(dict_list, key=lambda i: i[key], reverse=descending)


# https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
def find_most_frequent(List):
    """Finds most frequent element in a list"""
    return max(set(List), key=List.count)


#### ------------------------------------------------------------------------------------------ ####
####    DICTIONARIES: SEARCH & MANIPULATION
#### ------------------------------------------------------------------------------------------ ####


def create_dict_key(in_dict, key, value=0):
    """Adds key to dict if necessary, init'd with value"""
    if key not in in_dict.keys():
        in_dict[f"{key}"] = value
        return True
    return False


def increment_dict_key(in_dict, key, by_value=1):
    """Increments a dict key (default: by 1), if necessary, initializes it first"""
    if key not in in_dict.keys():
        create_dict_key(in_dict, key)
    in_dict[key] = in_dict[key] + by_value


def add_to_dict_key(in_dict, key, obj):
    """Adds an object to the list of a dict key, if necessary, initializes it first"""
    if key not in in_dict.keys():
        create_dict_key(in_dict, key, [])
    in_dict[key].append(obj)


def get_dict_value(in_dict, key, not_found_value=None):
    """Returns dictionary value if key it exists, else not_found_value.

    Args:
        in_dict ([dict of objects]): Input dict.
        key ([str]): The key in question
        not_found_value (object): Return value if key was not found.
    """
    if key not in in_dict.keys():
        return not_found_value
    return in_dict[key]


#### ------------------------------------------------------------------------------------------ ####
####    DATE
#### ------------------------------------------------------------------------------------------ ####


def getTimeStamp(file_name_friendly=False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if file_name_friendly:
        ts = ts.replace(" ", "-")
        return remove_invalid_file_chars(ts)
    else:
        return ts


def get_date():
    """Get today date object."""
    return datetime.today()


#### ------------------------------------------------------------------------------------------ ####
####    MULTITHREADING
#### ------------------------------------------------------------------------------------------ ####


def run_multithreaded(func, args, num_workers=10, show_progress=True):
    """Runs a particular function in multithread mode with a pool of num_workers workers.

    Args:
        func ([type]): function to run in multithreaded mode
        args ([type]): arguments as a list/generator, must be tuples when more than one. E.g.:
                       args = [(i, f) for i, f in enumerate(file_list)] yields a list with (index, file) entries
        num_workers (int, optional): Thread pool number of workers. Defaults to 10.
        show_progress (bool, optional): tqdm progress. Defaults to True.

    Returns:
        [type]: Thread results as a list (None entries if nothing is returned).
    """

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(executor.map(lambda p: func(*p), args), total=len(args))
            )
        else:
            results = list(executor.map(lambda p: func(*p), args))

    return results


#### ------------------------------------------------------------------------------------------ ####
####    MISCELLANEOUS
#### ------------------------------------------------------------------------------------------ ####


# ONLY works for Python 3.8, disable for now
# def get_var_name(any_var):
#     """ Gets variable name as string. """
#     return f'{any_var=}'.split('=')[0]


# TODO: https://stackoverflow.com/questions/56437081/using-tqdm-with-subprocess
def exec_shell_command(command, print_output=False, silent=False):
    """Executes a shell command using the subprocess module.
    command: standard shell command (SPACE separated - escape paths with '' or "")
    print_output: output command result to console
    returns: list containing all shell output lines
    """
    if not silent:
        print(f"Exec shell command '{command}'")
    # pre 2021:
    # regex = r"[^\s\"']+|\"([^\"]*)\"|'([^']*)'"
    # command_list = get_regex_match_list(command, regex)

    # 2021: simplified - maybe breaks some older calls
    # WARNING: use shlex.quote with strings containing user input
    # (cf. https://stackoverflow.com/questions/54581349/python-subprocess-ignores-backslash-in-convert-command)
    command_list = shlex.split(command)

    # DEBUG
    # print(command_list)

    process = subprocess.Popen(
        command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    output = []
    # execute reading output line by line
    for line in iter(process.stdout.readline, b""):
        line = line.decode(sys.stdout.encoding)
        output.append(line.replace("\n", ""))
        if print_output:
            sys.stdout.write(line)
    return output


def update_config_file(cfg_path, update_dict):
    if not bool(update_dict):
        # empty update dict
        return

    cfg = read_json(cfg_path)
    now = datetime.now()
    if not cfg:
        cfg = {"created": now.strftime("%Y-%m-%d, %H:%M:%S")}
    else:
        cfg["updated"] = now.strftime("%Y-%m-%d, %H:%M:%S")

    print(f"Update config: {cfg_path}")
    # make sure not to overwrite attributes here
    for k, v in update_dict.items():
        cfg[k] = v
        print(f"{k} => {v}")

    write_json(cfg_path, cfg, True)


# Advanced info about video (requires FFMPG!!)
# src: https://stackoverflow.com/questions/45738856/python-how-to-get-display-aspect-ratio-from-video
def get_video_info(video_file):
    """Advanced info about video.

    Args:
        video_file (string): Path to video file.

    Returns:
        dict: {
            duration,
            width,
            height,
            display_width,
            display_height,
            display_aspect_ratio,
            storage_aspect_ratio,
            pixel_aspect_ratio}
    """
    cmd = (
        'ffprobe -i "{}" -v quiet -print_format json -show_format -show_streams'.format(
            video_file
        )
    )
    jsonstr = None
    try:
        jsonstr = subprocess.check_output(cmd, shell=True, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info for {video_file}:")
        print(e.output)
        return None

    r = json.loads(jsonstr)
    # look for "codec_type": "video". take the 1st one if there are mulitple
    video_stream_info = [x for x in r["streams"] if x["codec_type"] == "video"][0]

    width, height = int(video_stream_info["width"]), int(video_stream_info["height"])
    duration = float(video_stream_info["duration"])

    if (
        "display_aspect_ratio" in video_stream_info
        and video_stream_info["display_aspect_ratio"] != "0:1"
    ):
        a, b = video_stream_info["display_aspect_ratio"].split(":")
        dar = int(a) / int(b)
    else:
        # some videos do not have the info of 'display_aspect_ratio'
        dar = width / height
        ## not sure if we should use this
        # cw,ch = video_stream_info['coded_width'], video_stream_info['coded_height']
        # sar = int(cw)/int(ch)
    if (
        "sample_aspect_ratio" in video_stream_info
        and video_stream_info["sample_aspect_ratio"] != "0:1"
    ):
        # some video do not have the info of 'sample_aspect_ratio'
        a, b = video_stream_info["sample_aspect_ratio"].split(":")
        sar = int(a) / int(b)
    else:
        sar = dar
    par = dar / sar
    ret = {
        "duration": duration,
        "width": width,
        "height": height,
        # square pixel dims (see: https://stackoverflow.com/questions/51825784/output-image-with-correct-aspect-with-ffmpeg)
        "display_width": int(width * sar),
        "display_height": height,
        "dar": dar,  # display aspect ratio
        "sar": sar,  # storage aspect ratio
        "par": par,  # pixel aspect ratio
    }
    return ret


class switch(object):
    """This class provides switch functionality.

    USAGE (also works with enums):
    ------------------------------

        thevar = "test"
        for case in switch(thevar):
            if case("test"):
                break
            if case():
                break
    """

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
