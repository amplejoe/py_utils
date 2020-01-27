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

# @Author: Andy <amplejoe>
# @Date:   2019-07-27
# @Email:  aleibets@itec.aau.at
# @Filename: utils.py
# @Last modified by: aleibets
# @Last modified time: 2020-01-27T12:08:47+01:00
# @description:  Utility class for common python tasks
# @notes:  requires ntpath, natsort, shutil

"""
    title			:utils.py
    description     :Utility class for common python tasks
    author			:Andreas Leibetseder (aleibets@itec.aau.at)
    version         :1.0
    usage			:import module and use contained functions
    notes			:requirements: natsort, shutil
    python_version	:3.6
    ===========================================================================
"""

# # IMPORTS
import os
import sys
import errno
import json
from datetime import datetime
import shutil
from natsort import natsorted
import pathlib
import re
import subprocess


# # USER INPUT RELATED


def confirm():
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("[y/n]").lower()
    return answer == "y"


def confirm_delete_file(path):
    p = to_path(path)
    if p.is_file():
        print("File exists: %s, delete file?" % (path))
        if (confirm()):
            remove_file(path)


def confirm_delete_path(path):
    p = to_path(path)
    if p.is_dir():
        print("Path exists: %s, delete folder?" % (path))
        if (confirm()):
            remove_dir(path)
        else:
            # user explicitly typed 'n'
            return False
    return True


def confirm_overwrite(path):
    p = to_path(path)
    confirmed = False
    if p.is_dir():
        confirmed = confirm_delete_path(path)
    elif p.is_file():
        confirmed = confirm_delete_file(path)
    else:
        confirmed = True
    if not confirmed:
        return False
    return make_dir(path, True)


# # FILE OPERATIONS

def exists_dir(*p):
    """ Checks whether a directory really exists.
    """
    return to_path(*p).is_dir()


def exists_file(*p):
    """ Checks whether a file really exists.
    """
    return to_path(*p).is_file()


def to_path(*p):
    """ Convert string to pathlib path.
    """
    return pathlib.Path(*p)


def to_path_str(*p):
    """ Convert string to pathlib path.
        Returns string representation.
    """
    return to_path(*p).as_posix()


def join_paths(path, *paths):
    """ Joins path with arbitrary amount of other paths.
    """
    return to_path(path).joinpath(to_path(*paths))


def join_paths_str(path, *paths):
    """ Joins path with arbitrary amount of other paths.
        Returns string representation.
    """
    return join_paths(path, *paths).as_posix()


def make_dir(path, show_info=False, overwrite=False):
    """ Creates a directory
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
        if (overwrite):
            remove_dir(path)
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Unexpected error: %s", str(e.errno))
            raise  # This was not a "directory exists" error..
        # print("Directory exists: %s", path)
        return False
    if (show_info):
        print(f"Created dir: {path}")
    return True


def remove_dir(path):
    exists_dir = os.path.isdir(path)
    if exists_dir:
        # os.rmdir(path) # does not work if not empty
        shutil.rmtree(path, ignore_errors=True)  # ignore errors on windows


def get_directories(directory):
    retList = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            # print os.path.join(root, filename)
            retList.append(os.path.abspath(os.path.join(root, dir)))
    return retList


def get_immediate_subdirs(a_dir, full_path=True):
    """ Returns a list of immediate sub-directories of a path.
        full_path (Default: True): get full path or sub-dir names only
    """
    path = to_path(a_dir)
    if not path.is_dir():
        return []
    if full_path:
        return [f.as_posix() for f in path.iterdir() if f.is_dir()]
    else:
        return [f.stem for f in path.iterdir() if f.is_dir()]


def get_nth_parentdir(file_path, n=0, full_path=False):
    """ Get nth parent directory of a file path, starting from the back (file).
        (Default: 0, i.e. the first directory after a potential filename)
        full_path: return full path until nth parent dir
    """
    p = to_path(file_path)
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
    """file path only (strips file from its path)
    """
    p = to_path(file_path)
    return p.parents[0].as_posix()


def get_file_paths(directory, *extensions):
    """ Get all file paths of a directory (optionally with file extensions
        usage example: get_file_paths("/mnt/mydir", ".json", ".jpg")
        changes:
            - 2019: using pathlib (python3) now
    """
    dir = to_path(directory)

    all_files = []
    for currentFile in dir.glob('**/*'):
        if not currentFile.is_file():
            continue
        fext = currentFile.suffix
        if extensions and fext not in extensions:
            continue
        # as_posix: Return a string representation
        # of the path with forward slashes (/)
        all_files.append(currentFile.as_posix())
    return all_files


def path_to_relative_path(path, relative_to_path):
    """ Return sub-path relative to input path
    """
    path = to_path(path)
    rel_to = to_path(relative_to_path)
    return path.relative_to(rel_to).as_posix()


def get_full_file_name(file_path):
    """full file name plus extension
    """
    return to_path(file_path).name


def get_file_name(file_path):
    """ Get file name of file
    """
    return to_path(file_path).stem


def get_file_ext(file_path):
    """ Get file extension of file
    """
    return to_path(file_path).suffix


def remove_file(path):
    to_path(path).unlink()


def copy_to(src_path, dst_path, follow_symlinks=True):
    """ Copies src_path to dst_path.
        If dst is a directory, a file with the same basename as src is created
        (or overwritten) in the directory specified.
    """
    shutil.copy(src_path, dst_path, follow_symlinks=follow_symlinks)


# # MISCELLANEOUS

def exec_shell_command(command, print_output=False):
    """ Executes a shell command using the subprocess module.
        command: standard shell command (SPACE separated)
        print_output: output command result to console
        returns: list containing all shell output lines
    """
    print(f"Exec shell command '{command}'")
    command_list = command.split(" ")

    process = subprocess.Popen(
        command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    output = []
    # execute reading output line by line
    for line in iter(process.stdout.readline, b''):
        line = line.decode(sys.stdout.encoding)
        output.append(line.replace("\n", ""))
        if (print_output):
            sys.stdout.write(line)
    return output


def get_attribute_from(file_name, attribute):
    """ gets an attribute from a file name as string
        format: a1_VAL1_a2_VAL2_a3_VAL3.EXT
        Info: VALs cannot contain '_'; '.EXT' is optional
    """
    split_one = file_name.split(f"{attribute}_")[1]
    split_two = split_one.split("_")[0]
    return split_two


def avg_list(lst):
    """Average value of a list of values"""
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def create_dict_key(split_dict, key, value=0):
    """Adds key to dict if necessary, init'd with value"""
    if key not in split_dict.keys():
        split_dict[f'{key}'] = value


def safe_div(x, y):
    """ Zero safe division
    """
    if y == 0:
        return 0
    return x / y


def getTimeStamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_current_dir():
    """ Returns current working directory.
    """
    return os.getcwd()


def get_script_dir():
    """ Returns directory of currently running script.
    """
    return os.path.dirname(os.path.realpath(__file__))


def is_absolute_path(path):
    """ Checks if path is absolute (also for non-existing paths!).
    """
    return os.path.isabs(path)


def rel_to_abs_path(path):
    current_dir = os.getcwd()
    return join_paths_str(current_dir, path)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def filter_list_by_partial_word(word, list_to_filter):
    """ partially matches a word with a list and returnes filtered list
    """
    return list(filter(lambda x: word in x, list_to_filter))


def nat_sort_list(l):
    return natsorted(l, key=lambda y: y.lower())


def set_environment_variable(key, value):
    """ Sets an OS environment variable.
    """
    os.environ[key] = value


def sort_list_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)


def get_attribute_from_json(path, attr):
    """ gets attribute from json file
    """
    # get fps
    with open(path) as json_file:
        data = json.load(json_file)
        return data[attr]


def format_number(number, precision=3, width=3):
    opts = "{:%s.%sf}" % (str(width), str(precision))
    return opts.format(number)


class switch(object):
    """ This class provides switch functionality.
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
