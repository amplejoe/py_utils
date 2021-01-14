from typing import List, Set, Dict, Tuple, Optional
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

        file_types = None
        if files is not None:
            file_types = (files, ("all files", "*.*"))
        else:
            file_types = ("all files", "*.*")

        selected_file = filedialog.askopenfilename(
            initialdir=dir,
            title=title,
            filetypes=file_types,
        )
        return selected_file