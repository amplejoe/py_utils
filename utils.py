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
# @Date:   2019-07-27
# @Email:  aleibets@itec.aau.at
# @Filename: utils.py
# @Last modified by: aleibets
# @Last modified time: 2020-02-12T16:15:58+01:00
# @description:  Utility class for common python tasks
# @notes:  requires ntpath, natsort, shutil

"""
    title			:utils.py
    description     :Utility class for common python tasks
    author			:Andreas <aleibets>
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
import numpy as np


# # USER INPUT RELATED

def select_option(options, *, msg=None, default=None):
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
    if default < 0 or default > len(options)-1:
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
        accepted_answers.append("")  # alows users to press enter
        user_prompt = user_prompt.replace(default, default.upper())
    if msg is not None:
        print(msg)
    answer = None
    while answer not in accepted_answers:
        answer = input(user_prompt).lower()
        if answer == "":
            answer = default
    return (answer == "y")


def confirm_delete_file(path, default=None):
    p = to_path(path, as_string=False)
    if p.is_file():
        if (confirm("File exists: %s, delete file?" % (path), default)):
            remove_file(path)
        else:
            # user explicitly typed 'n'
            return False
    return True


def confirm_delete_path(path, default=None):
    p = to_path(path, as_string=False)
    if p.is_dir():
        if (confirm("Path exists: %s, delete folder?" % (path), default)):
            remove_dir(path)
        else:
            # user explicitly typed 'n'
            return False
    return True


def confirm_overwrite(path, default=None):
    """ Confirms overwriting a path or a file.
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
        # check if a dir needs to be created
        ext = get_file_ext(path)
        if ext == '':
            make_dir(path, True)
    return confirmed


# # FILE OPERATIONS

def exists_dir(*p):
    """ Checks whether a directory really exists.
    """
    return to_path(*p, as_string=False).is_dir()


def exists_file(*p):
    """ Checks whether a file really exists.
    """
    return to_path(*p, as_string=False).is_file()


def to_path_url(*p, as_string=True):
    """ Convert URL to pathlib path.
        INFO: Use this with URLS, as to_path will raise an error with URLS
    """
    pth = pathlib.Path(*p)
    if as_string:
        return pth.as_posix()
    else:
        return pth


def to_path_url_str(*p):
    """ Convert URL string to pathlib path.
        Returns string representation.
        ------
        deprecated - to_path_url already per default returns string
    """
    return to_path_url(*p)


def to_path(*p, as_string=True):
    """ Convert string to pathlib path.
        INFO: Path resolving removes stuff like ".." with 'strict=False' the
        path is resolved as far as possible -- any remainder is appended
        without checking whether it really exists.
    """
    pl_path = pathlib.Path(*p)
    ret = pl_path.resolve(strict=False) # default return in case it is absolute path

    if not pl_path.is_absolute():
        # don't resolve relative paths (pathlib makes them absolute otherwise)
        ret = pl_path
    
    if as_string:
        return ret.as_posix()
    else:
        return ret


def to_path_str(*p):
    """ Convert string to pathlib path.
        Returns string representation.
        --------
        deprecated - 'to_path' does the same with as_string=True
    """
    return to_path(*p)


def join_paths(path, *paths, as_string=True):
    """ Joins path with arbitrary amount of other paths.
    """
    joined = to_path(path, as_string=False).joinpath(to_path(*paths, as_string=False))
    joined_resolved = to_path(joined, as_string=False)
    if as_string:
        return joined_resolved.as_posix()
    else:   
        return joined_resolved


def join_paths_str(path, *paths):
    """ Joins path with arbitrary amount of other paths.
        Returns string representation.
        --------
        deprecated - 'join_paths' does the same with as_string=True
    """
    return join_paths(path, *paths)


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
    path = to_path(a_dir, as_string=False)
    if not path.is_dir():
        return []
    if full_path:
        return [f.as_posix() for f in path.iterdir() if f.is_dir()]
    else:
        return [get_nth_parentdir(f) for f in path.iterdir() if f.is_dir()]


def get_nth_parentdir(file_path, n=0, full_path=False):
    """ Get nth parent directory of a file path, starting from the back (file).
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
    """file path only (strips file from its path)
    """
    p = to_path(file_path, as_string=False)
    return p.parents[0].as_posix()


def get_file_paths(directory, *extensions):
    """ Get all file paths of a directory (optionally with file extensions
        usage example: get_file_paths("/mnt/mydir", ".json", ".jpg")
        changes:
            - 2019: using pathlib (python3) now
    """
    dir = to_path(directory, as_string=False)

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
    path = to_path(path, as_string=False)
    rel_to = to_path(relative_to_path, as_string=False)
    return path.relative_to(rel_to).as_posix()


def get_full_file_name(file_path):
    """full file name plus extension
    """
    return to_path(file_path, as_string=False).name


def get_file_name(file_path):
    """ Get file name of file
    """
    return to_path(file_path, as_string=False).stem


def get_file_ext(file_path):
    """ Get file extension of file
    """
    return to_path(file_path, as_string=False).suffix


def remove_file(path):
    p = to_path(path, as_string=False)
    if exists_file(p):
        p.unlink()


def copy_to(src_path, dst_path, follow_symlinks=True):
    """ Copies src_path to dst_path.
        If dst is a directory, a file with the same basename as src is created
        (or overwritten) in the directory specified.
    """
    shutil.copy(src_path, dst_path, follow_symlinks=follow_symlinks)

# # MEMORY


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
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# # Numpy specific

def is_np_arr_in_list(np_array, lst):
    """ Checks if a list of numpy arrays contains a specific numpy array (Identity).
    """
    return next((True for elem in lst if elem is np_array), False)


def remove_np_from_list(lst, np_array):
    """ Removes a numpy array from a list.
    """
    ind = 0
    size = len(lst)
    while ind != size and not np.array_equal(lst[ind], np_array):
        ind += 1
    if ind != size:
        lst.pop(ind)
    else:
        pass
        # raise ValueError('array not found in list.')

# # MISCELLANEOUS


def exit(msg=None):
    """ Exits script with optional message.
    """
    if (msg):
        print(f"{msg}")
    print("Exit script.")
    sys.exit()


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
    file_name = get_file_name(file_name)  # make sure file_name is no path
    split_one = file_name.split(f"{attribute}_")[1]
    split_two = split_one.split("_")[0]
    return split_two


def avg_list(lst):
    """Average value of a list of values"""
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def create_dict_key(dict, key, value=0):
    """Adds key to dict if necessary, init'd with value"""
    if key not in dict.keys():
        dict[f'{key}'] = value


def increment_dict_key(dict, key, by_value=1):
    """Increments a dict key (default: by 1), if necessary, initializes it first"""
    if key not in dict.keys():
        create_dict_key(dict, key)
    dict[key] = dict[key] + by_value


def add_to_dict_key(dict, key, object):
    """Adds an object to the list of a dict key, if necessary, initializes it first"""
    if key not in dict.keys():
        create_dict_key(dict, key, [])
    dict[key].append(object)


def safe_div(x, y):
    """ Zero safe division
    """
    if y == 0:
        return 0
    return x / y


def getTimeStamp(file_name_friendly=False):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file_name_friendly:
        ts = ts.replace(" ", "-")
        return remove_invalid_file_chars(ts)
    else:
        return ts


def get_date():
    """ Get today date object.
    """
    return datetime.today()


def get_date_str():
    """ Get today's date as string in form of YYYY/MM/DD
    """
    today = get_date()
    return today.strftime("%Y/%m/%d")


def get_current_dir():
    """ Returns current working directory.
    """
    return to_path(pathlib.Path.cwd())


def get_script_dir():
    """ Returns directory of currently running script.
    """
    return to_path(os.path.dirname(os.path.realpath(__file__)))


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


def is_partial_word_in_list(word, lst):
    res = filter_list_by_partial_word(word, lst)
    return len(res) > 0


def filter_list_by_partial_word(word, list_to_filter):
    """ partially matches a word with a list and returns filtered list
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


# https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
def find_most_frequent(List):
    """ Finds most frequent element in a list
    """
    return max(set(List), key=List.count)


def read_json(path):
    """ Loads a json file into a variable.
        Parameters:
        ----------
        path : str
            path to json file
        Return : dict
            dict containing all json fields and attributes
    """
    data = None
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def write_json(path, data):
    """ Writes a json dict variable to a file.
        Parameters:
        ----------
        path : str
            path to json output file
        data : dict
            data compliant with json format
    """
    with open(path, 'w', newline='') as outfile:
        json.dump(data, outfile)


def read_json_arr(json_path):
    """ Load json as array of lines
    """
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def get_attribute_from_json(path, attr):
    """ gets attribute from json file
    """
    return read_json(path)[attr]


def read_file_to_array(path):
    """ Reads all lines of a file into an array.
    """
    arr = None
    with open(path) as file:
        arr = file.readlines()
    return arr


def write_string_to_file(str_to_write, file_path):
    with open(file_path, 'w', newline='') as file:
        file.write(str_to_write)


def find_similar_folder(target_path, folder_list):
    """ Finds similar subfolder name in a directory from a list of names.
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
    """ Prompts user to confirm or enter a folder name.
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
        while sim_folder not in all_subdirs:
            sim_folder = input(f"Please enter {name} dir: ")
            if sim_folder not in all_subdirs:
                print("Dir not found, please try again (Ctrl+C to quit).")
    print(f"Using {sim_folder} ({name}).")
    return join_paths_str(target_path, sim_folder)


def format_number(number, precision=3, width=3):
    opts = "{:%s.%sf}" % (str(width), str(precision))
    return opts.format(number)


def remove_invalid_file_chars(input):
    """
    Removes invalid chars (Windows) from string.
    """
    invalid = '<>:"/\\|?* '
    for char in invalid:
        input = input.replace(char, '')
    return input


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
