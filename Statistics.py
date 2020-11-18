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
# @Date: 2019-09-10T14:16:08+02:00
# @Filename: Statistics.py
# @Last modified by: aleibets
# @Last modified time: 2019-09-11T16:17:36+02:00
# @description:

from pandas import DataFrame


class Statistics:
    """Class for numerical statistics saving, adding, printing """

    def __init__(self, col_labels, *, index_column_name="", is_numeric=True):
        self.row_header_label = index_column_name  # name for row headers (index column)
        self.value_cols = col_labels  # only cols holding values
        self.stat_dict = {}
        self.is_numeric = is_numeric

    def get_stat_df(self):
        df = DataFrame.from_dict(
            self.stat_dict, columns=self.value_cols, orient="index"
        )
        return df

    def init_row(self, row_name):
        """Initializes a new row iff not already existing"""
        # set all cols to zero
        if row_name not in self.stat_dict.keys():
            initial_values = [0] * len(self.value_cols)
            if not self.is_numeric:
                initial_values = [""] * len(self.value_cols)
            self.stat_dict[f"{row_name}"] = initial_values

    def add_single(self, row_name, col_name, value):
        """ Add a single entry to internal dict"""
        self.init_row(row_name)
        idx = self.value_cols.index(col_name)
        self.stat_dict[f"{row_name}"][idx] = self.stat_dict[f"{row_name}"][idx] + value

    def add_row(self, row_name, row_values=[]):
        """Add a whole entry (row) to internal dict"""
        self.init_row(row_name)
        cnt = 0
        for val in row_values:
            self.stat_dict[f"{row_name}"][cnt] = (
                self.stat_dict[f"{row_name}"][cnt] + val
            )
            cnt = cnt + 1

    def write_stat_csv(self, out_path):
        df = self.get_stat_df()
        df.to_csv(out_path, index=True, index_label=self.row_header_label, header=True)
        print(f"Wrote {out_path}")
