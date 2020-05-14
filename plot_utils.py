import seaborn as sns
import matplotlib.pyplot as plt
from py_utils import utils
sns.set(style="whitegrid")


def barplot(df):
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="day", y="total_bill", data=tips)


# orig src: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-mytrainer-py
def plot_train_val_loss(model_folder):
    experiment_metrics = utils.read_json_arr(model_folder + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics],
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.show()
