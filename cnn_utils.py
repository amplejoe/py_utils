from tqdm import tqdm
from . import utils
import copy
import itertools
import functools
import cv2
import numpy as np
import math
import csv

# detectron2 imports
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

DEFAULT_IMG_DIRS = ['img', 'imag', 'frame', 'pic', 'phot']
DEFAULT_IMG_EXT = ['.jpeg', '.jpg', '.png', '.bmp']
PIXEL_MEAN_FILE = "pixel_means_train.csv"


# pass a list of dicts (default)
# OR pass dict: list(product_dict(**mydict))
# src: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def cart_product_dict(**kwargs):
    """ produces a list of cartesian product dicts.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def dump_cfg(cfg):
    if not utils.exists_dir(cfg.OUTPUT_DIR):
        print("Cannot dump cfg, dir does not exists!")
        return
    cfg_out = utils.join_paths_str(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_out, "w") as co:
        co.write(cfg.dump())


def params_list_to_dict(params):
    """ Creates {D2_param -> [v1, v2, ...]}
    """
    params_dict = {}
    for item in params:
        # evaluate items if 'eval' flag is set
        if is_key_enabled(item, "eval"):
            item["values"] = [eval(i) for i in item["values"]]
        params_dict[item["detectron_field"]] = item["values"]
    return params_dict


def is_key_set(dict, key):
    return key in dict and dict[key] is not None


def is_key_enabled(dict, key):
    return key in dict and dict[key]


# src: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def get_d2_cfg_attr(cfg, attr, *args):
    def _getattr(cfg, attr):
        return getattr(cfg, attr, *args)
    return functools.reduce(_getattr, [cfg] + attr.split('.'))


# src: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def set_d2_cfg_attr(cfg, setting, evaluate=False):
    """ Recudsively set attribute for d2 cfg
    """
    field = setting["detectron_field"]
    value = setting["value"]
    if evaluate:
        value = eval(value)
    pre, _, post = field.rpartition('.')
    return setattr(get_d2_cfg_attr(cfg, pre) if pre else cfg, post, value)


def register_datasets(ds_info):
    """ WARNING: don't do this multiple times
    """
    register_coco_instances(f"{ds_info['ds_name']}_train", {}, ds_info["ds_train"], ds_info['image_path'])
    register_coco_instances(f"{ds_info['ds_name']}_val", {}, ds_info["ds_val"], ds_info['image_path'])
    register_coco_instances(f"{ds_info['ds_name']}_test", {}, ds_info["ds_test"], ds_info['image_path'])


def load_d2_cfg(config_file):
    config = get_cfg()
    config.merge_from_file(
        config_file
    )
    return config


# based on https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
def calc_pixel_mean_std(ds_info, img_ext=DEFAULT_IMG_EXT, num_channels=3):
    """ Calculates the pixel mean of the training dataset in BGR format.
        Parameters
        ----------
        ds_info : dict
            contains property 'img_path', which is the DS image root
        img_ext : list of str
            image extensions as a list, e.g. [".jpg", ".png"]
        Return : tuple of list of float
            (list) bgr_mean, (list) bgr_std
    """

    # load potentially saved DS mean file
    mean_file_url = utils.join_paths_str(ds_info['ds_path'], PIXEL_MEAN_FILE)
    if utils.exists_file(mean_file_url):
        with open(mean_file_url, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            headers = next(csv_reader)
            data_rows = [row for row in csv_reader]
            data_dict = {}
            for row in data_rows:
                data = [float(x) for i, x in enumerate(row) if i > 0]
                data_dict[row[0]] = data
            print(f"Loaded pixel means from file: {mean_file_url}")
            return data_dict["color_mean"], data_dict["std_mean"]

    # img_paths = utils.get_file_paths(ds_info['image_path'], *img_ext)
    train_img_paths = [utils.join_paths_str(ds_info['image_path'], x["file_name"]) for x in ds_info["train_images_json"]]
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)

    pixel_num = 0  # store all pixel number in the dataset
    for file in tqdm(train_img_paths, desc="Calculating DS pixel means"):
        im = cv2.imread(file)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.0
        pixel_num += (im.size/num_channels)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean_normalized = channel_sum / pixel_num
    bgr_std_normalized = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean_normalized))

    # unnormalize ()
    bgr_mean = bgr_mean_normalized * 255.0
    bgr_std = bgr_std_normalized * 255.0

    # change the format from bgr to rgb
    # rgb_mean = list(bgr_mean)[::-1]
    # rgb_std = list(bgr_std)[::-1]

    color_mean = [float(round(x, 3)) for x in list(bgr_mean)]
    std_mean = [float(round(x, 3)) for x in list(bgr_std)]

    with open(mean_file_url, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['type', 'b', 'g', 'r'])
        csv_writer.writerow(['color_mean'] + color_mean)
        csv_writer.writerow(['std_mean'] + std_mean)
        print(f"Saved pixel means: {mean_file_url}")

    # limit to 3 decimals
    return color_mean, std_mean


def calc_min_max_pixel_vals(ds_info, in_img_ext=DEFAULT_IMG_EXT, num_channels=3):
    """ Calculates the pixel min and max pixel values of the training dataset in BGR format.
        Parameters
        ----------
        ds_info : dict
            contains property 'img_path', which is the DS image root
        img_ext : list of str
            image extensions as a list, e.g. [".jpg", ".png"]
        Return : tuple of list of float
            (list) min_px_values, (list) max_px_values
    """

    # img_paths = utils.get_file_paths(ds_info['image_path'], *in_img_ext)
    train_img_paths = [utils.join_paths_str(ds_info['image_path'], x["file_name"]) for x in ds_info["train_images_json"]]
    max_total = np.zeros((num_channels, 1))
    min_total = np.zeros((num_channels, 1))
    for file in tqdm(train_img_paths, desc="Calculating min max values"):
        image = cv2.imread(file)
        # max, min channel values
        # tmp = [0] * num_channels
        bgr_tuple = cv2.split(image)  # b, g, r
        max_tmp = list([np.amax(x) for x in bgr_tuple])
        min_tmp = list([np.amin(x) for x in bgr_tuple])
        for i, val in enumerate(max_tmp):
            max_total[i] = max_tmp[i] if max_total[i] < max_tmp[i] else max_total[i]
        for i, val in enumerate(min_tmp):
            min_total[i] = min_tmp[i] if min_total[i] > min_tmp[i] else min_total[i]
    return list(min_total.flatten()), list(max_total.flatten())


# # wrongly calcs std (over all images instead of over all pixels)
# def calc_pixel_mean_std_wrong(ds_info, num_channels=3):
#     """ Calculates the pixel mean of the input dataset in BGR format.
#         Return : tuple(list[bgr_mean], list[bgr_std])
#     """
#     img_paths = utils.get_file_paths(ds_info['image_path'], *IN_IMG_EXT)
#     count = 0
#     mean_total = np.zeros((num_channels, 1))
#     std_total = np.zeros((num_channels, 1))
#
#     for file in tqdm(img_paths, desc="Calculating pixel mean"):
#         image = cv2.imread(file)
#         # BGR
#         mean_img, std_img = cv2.meanStdDev(image)
#         mean_total += mean_img
#         std_total += std_img
#         count += 1
#
#     mean_total /= count
#     std_total /= count
#
#     return list(mean_total.flatten()), list(std_total.flatten())

def iter_to_epoch(iteration, data_size, batch_size):
    one_epoch = data_size / batch_size
    return iteration / one_epoch


def get_last_n_metrics(model_folder, n=10):
    experiment_metrics = utils.read_json_arr(model_folder + '/metrics.json')
    return experiment_metrics[-n:]


def create_d2_cfgs(ds_info, script_dir):
    """ Get detectron2 configs for a specific dataset. Returns list of cfgs depending on parameters
        Parameters
        ----------

        script_dir : str
            path of executing script
    """

    coco_ds = utils.read_json(ds_info["ds_train"])
    num_classes = len(coco_ds["categories"])

    # print(f"#images: {num_train_images}")

    # ds pixel mean, pixel std
    px_mean, px_std = calc_pixel_mean_std(ds_info)

    # get cartesian product of all parameters
    param_permuts = list(cart_product_dict(**ds_info["cfg"]["params"]))

    cnn_cfgs = {}
    for cnn in ds_info["cfg"]["cnns"]:
        cnn_cfgs[cnn["name"]] = []
        base_cfg = get_cfg()

        # # BASE YAML configs
        # D2 config files
        # (custom cfgs should be last in list as not to override settings by d2 base cfgs)
        if is_key_set(cnn, "cfg_urls"):
            for c in cnn["cfg_urls"]:
                base_cfg.merge_from_file(
                    utils.join_paths_str(script_dir, c)
                )

        base_cfg.DATASETS.TRAIN = (f"{ds_info['ds_name']}_train",)
        base_cfg.DATASETS.TEST = (f"{ds_info['ds_name']}_val", )
        #  D2 weights
        if is_key_set(cnn, "weight_url"):
            base_cfg.MODEL.WEIGHTS = cnn["weight_url"]
        # base_cfg.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, good enough for a toy dataset
        base_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # set pixel mean and std
        base_cfg.MODEL.PIXEL_MEAN = px_mean
        base_cfg.MODEL.PIXEL_STD = px_std

        # set max_iter to resemple ds_info['cfg']['training']['num_epochs']
        one_epoch = ds_info["num_train_images"] / base_cfg.SOLVER.IMS_PER_BATCH
        base_cfg.SOLVER.MAX_ITER = int(one_epoch * ds_info['cfg']['training']['num_epochs'])
        # validation period (validate after after 'validation_perc' * ITERS_PER_EPOCH)
        # INFO: every 20 iterations an entry is made to 'metrics.json', so be sure to set EVAL_PERIOD to a multiple of 20
        # (see: https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py#L186)
        log_interval = 20
        eval_period = int(one_epoch * ds_info["cfg"]["training"]["validation_perc"] / log_interval) * log_interval
        eval_period = log_interval if eval_period < 20 else eval_period
        base_cfg.TEST.EVAL_PERIOD = eval_period

        for perm in param_permuts:
            # create configs
            cfg = copy.deepcopy(base_cfg)

            out_dir_name = ""
            for d2_key, val in perm.items():
                set_d2_cfg_attr(cfg, {"detectron_field": d2_key, "value": val})
                parts = d2_key.split(".")
                field = parts[1] if len(parts) > 1 else parts[0]
                out_dir_name += f"{field}_{val}" if out_dir_name == "" else f"_{field}_{val}"

            out_root = ds_info["cfg"]["training"]["output_root"]
            out_path = utils.join_paths_str(ds_info["ds_path"], out_root, cnn["name"], out_dir_name)
            cfg.OUTPUT_DIR = out_path

            # checkpointing (save checkpoint after 'checkpoint_perc' * MAX_ITER)
            cfg.SOLVER.CHECKPOINT_PERIOD = int(math.ceil(cfg.SOLVER.MAX_ITER * ds_info["cfg"]["training"]["checkpoint_perc"]))

            cnn_cfgs[cnn["name"]].append(cfg)
    return cnn_cfgs


def get_ds_info(ds_path, ds_cfg):
    """ Creat dataset info dict with various fields used within cnn_utils.
        Parameters
        ----------
        ds_path : str
            path to dataset root (should contain COCO files)
        ds_cfg : dict
            dict containing individual DS settings, as accessed below
        return : dict
            dataset info

    """
    ds_name = utils.get_nth_parentdir(ds_path)
    dataset_info = {}
    dataset_info["cfg"] = ds_cfg
    dataset_info["ds_name"] = ds_name
    dataset_info["ds_path"] = ds_path
    dataset_info["ds_full"] = utils.join_paths_str(ds_path, dataset_info["cfg"]["coco"]["full"])
    dataset_info["ds_train"] = utils.join_paths_str(ds_path, dataset_info["cfg"]["coco"]["train"])
    dataset_info["ds_val"] = utils.join_paths_str(ds_path, dataset_info["cfg"]["coco"]["val"])
    dataset_info["ds_test"] = utils.join_paths_str(ds_path, dataset_info["cfg"]["coco"]["test"])
    dataset_info["image_path"] = utils.prompt_folder_confirm(dataset_info["ds_path"], DEFAULT_IMG_DIRS, 'images')
    dataset_info["num_total_images"] = len(utils.get_file_paths(dataset_info["image_path"], *DEFAULT_IMG_EXT))
    dataset_info["train_images_json"] = utils.get_attribute_from_json(dataset_info["ds_train"], "images")
    dataset_info["num_train_images"] = len(dataset_info["train_images_json"])
    return dataset_info
