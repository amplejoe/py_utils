import seaborn as sns

# Data science tools
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from PIL import Image

from defs.config import *
from defs.pytorch_transform import image_transforms
from . import utils

def data_folder_stats():

    # cur_dir = utils.get_current_dir() + "/"
    traindir = TRAIN_DIR
    validdir = VAL_DIR
    testdir = TEST_DIR

    # Empty lists
    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # Iterate through each category
    for d in os.listdir(traindir):
        categories.append(d)
        # Number of each image
        train_imgs = os.listdir(traindir + d)
        valid_imgs = os.listdir(validdir + d)
        test_imgs = os.listdir(testdir + d)
        n_train.append(len(train_imgs))
        n_valid.append(len(valid_imgs))
        n_test.append(len(test_imgs))

        # Find stats for train images
        for i in tqdm(train_imgs, desc="loading images (%s)" % (d)):
            img_categories.append(d)
            img = Image.open(traindir + d + '/' + i)
            img_array = np.array(img)
            # Shape
            hs.append(img_array.shape[0])
            ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid, 'n_test': n_test}).\
        sort_values('category')

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })

    cat_df.sort_values('n_train', ascending=False, inplace=True)
    return cat_df, image_df
    # cat_df.head()
    # cat_df.tail()

def show_folder_statistics(df):
    df.set_index('category')['n_train'].plot.bar(
    color='r', figsize=(20, 6))
    plt.xticks(rotation=80)
    plt.ylabel('Count')
    plt.title('Training Images by Category')
    plt.show()


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


def show_transforms_of(image, split):
    t = image_transforms[split]
    plt.figure(figsize=(24, 24))

    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        _ = imshow_tensor(t(image), ax=ax)
    plt.tight_layout()
    plt.show()
