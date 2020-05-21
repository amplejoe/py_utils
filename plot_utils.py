import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from . import utils
from . import cnn_utils

# import seaborn as sns
# sns.set(style="whitegrid")
matplotlib.use('TkAgg')  # necessary to prevent plots not showing,
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
def save_train_val_loss(model_folder, num_train_images, batch_size, out_file="train_val_loss.jpg", show=False):
    experiment_metrics = utils.read_json_arr(model_folder + '/metrics.json')

    # change fig size before 'plot'
    plt.figure(figsize=(4, 3), dpi=300, num=out_file).gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.title(out_file)
    # plt.figure(num=out_file)
    # fig = plt.figure()
    # fig.canvas.set_window_title(out_file)
    # plot (epochs instead of just iterations)
    plt.plot(
        [cnn_utils.iter_to_epoch(x['iteration'], num_train_images, batch_size) for x in experiment_metrics],
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [cnn_utils.iter_to_epoch(x['iteration'], num_train_images, batch_size) for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['training', 'validation'], loc='upper right')
    out_path = utils.join_paths_str(model_folder, out_file)
    plt.xlabel("epoch")
    plt.ylabel("loss (smooth-L1)")
    plt.tight_layout()  # prevent x/y labels to be cut off
    plt.savefig(out_path)
    if show:
        plt.show()
