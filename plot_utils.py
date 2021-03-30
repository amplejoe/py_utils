#!/usr/bin/env python

###
# File: plot_utils.py
# Created: Tuesday, 12th May 2020 5:46:35 pm
# Author: Andreas (amplejoe@gmail.com)
# -----
# Last Modified: Tuesday, 30th March 2021 2:13:58 am
# Modified By: Andreas (amplejoe@gmail.com)
# -----
# Copyright (c) 2021 Klagenfurt University
#
###

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from . import utils
from . import cnn_utils

# import seaborn as sns
# sns.set(style="whitegrid")

# TODO: check - is still needed on linux?
# matplotlib.use('TkAgg')  # necessary to prevent plots not showing,
# in case some import sets mode to 'agg' (e.g. detectron2)


# def barplot(df):
#     tips = sns.load_dataset("tips")
#     ax = sns.barplot(x="day", y="total_bill", data=tips)


def test_plot():
    X = np.linspace(-np.pi, np.pi, 256)
    C, S = np.cos(X), np.sin(X)

    plt.plot(X, C)
    plt.plot(X, S)
    plt.show()


# orig src: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-mytrainer-py
def save_train_val_loss(
    model_folder,
    num_train_images,
    batch_size,
    out_file="train_val_loss.jpg",
    plot_epoch=True,
    show=False,
):
    experiment_metrics = utils.read_json_arr(model_folder + "/metrics.json")

    # change fig size before 'plot'
    # (force integer with MxNLocator)
    plt.figure(figsize=(4, 3), dpi=300, num=out_file).gca().xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    total_x_axis = [
        cnn_utils.iter_to_epoch(x["iteration"], num_train_images, batch_size)
        for x in experiment_metrics
        if "total_loss" in x
    ]
    val_x_axis = [
        cnn_utils.iter_to_epoch(x["iteration"], num_train_images, batch_size)
        for x in experiment_metrics
        if "validation_loss" in x
    ]

    if not plot_epoch:
        total_x_axis = [x["iteration"] for x in experiment_metrics]
        val_x_axis = [
            x["iteration"] for x in experiment_metrics if "validation_loss" in x
        ]

    # plot
    plt.plot(
        total_x_axis, [x["total_loss"] for x in experiment_metrics if "total_loss" in x]
    )
    plt.plot(
        val_x_axis,
        [x["validation_loss"] for x in experiment_metrics if "validation_loss" in x],
    )
    plt.legend(["training", "validation"], loc="upper right")
    out_path = utils.join_paths_str(model_folder, out_file)
    plt.xlabel("epoch" if plot_epoch == True else "iteration")
    plt.ylabel("loss (smooth-L1)")
    plt.tight_layout()  # prevent x/y labels from being cut off
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.clf()
