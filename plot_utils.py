import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from . import utils
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
def save_train_val_loss(model_folder, show=False):
    experiment_metrics = utils.read_json_arr(model_folder + '/metrics.json')

    # change fig size before 'plot'
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(
        [x['iteration'] for x in experiment_metrics],
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    out_path = utils.join_paths_str(model_folder, "train_val_loss.jpg")
    plt.savefig(out_path)
    if show:
        plt.show()
