#!/usr/bin/env python

###
# File: GUITools.py
# Created: Thursday, 14th January 2021 12:37:16 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:13:14 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###

from typing import Tuple
import tkinter as tk
from tkinter import filedialog
from . import utils


class GUITools:
    def __init__(self):
        root = tk.Tk()
        root.withdraw()

    def pick_file(
        self,
        files: Tuple[str, str],
        title: str = "Select file",
        dir: str = utils.get_current_dir(),
    ) -> str:
        """Lets user pick a file of a certain type.

        Args:
            init_dir (str): Initial directory of file picker
            files (tuple[str, str]): File types to pick, e.g. ("video files", "*.mp4 *.avi")

        Returns:
            str: picked file path
        """
        root = tk.Tk()
        root.withdraw()

        file_types = ("all files", "*.*")
        if files is not None:
            file_types = (files, ("all files", "*.*"))

        selected_file = filedialog.askopenfilename(
            initialdir=dir,
            title=title,
            filetypes=file_types,
        )
        return selected_file