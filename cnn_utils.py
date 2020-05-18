from tqdm import tqdm
from . import utils
import copy
import itertools
import functools
import cv2
import numpy as np

# detectron2 imports
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

SETTINGS_FILE = "settings.json"
IN_IMG_FOLDERS = ['img', 'imag', 'frame', 'pic', 'phot']
IN_IMG_EXT = ['.jpeg', '.jpg', '.png', '.bmp']

COCO = "coco.json"
COCO_TRAIN = "coco_train.json"
COCO_VAL = "coco_val.json"
COCO_TEST = "coco_test.json"
GLOBAL_SETTINGS = "config.json"

OUT_DIR_ROOT = "models"


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


def load_d2_cfg(ds_info, config_file):
    config = get_cfg()
    config.merge_from_file(
        config_file
    )
    return config


# based on https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
def calc_pixel_mean_std(ds_info, num_channels=3):
    """ Calculates the pixel mean of the input dataset in BGR format.
        Return : tuple(list[bgr_mean], list[bgr_std])
    """
    img_paths = utils.get_file_paths(ds_info['image_path'], *IN_IMG_EXT)
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)

    pixel_num = 0  # store all pixel number in the dataset
    for file in tqdm(img_paths, desc="Calculating DS pixel mean"):
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

    # limit to 3 decimals
    return ([float(round(x, 3)) for x in list(bgr_mean)]), [float(round(x, 3)) for x in list(bgr_std)]


def calc_min_max_pixel_vals(ds_info, num_channels=3):
    img_paths = utils.get_file_paths(ds_info['image_path'], *IN_IMG_EXT)
    max_total = np.zeros((num_channels, 1))
    min_total = np.zeros((num_channels, 1))
    for file in tqdm(img_paths, desc="Calculating min max values"):
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


def create_d2_cfgs(ds_info, d2_configs_root, cnns, parameters):
    """ Get detectron2 configs for a specific dataset. Returns list of cfgs depending on parameters
    """

    coco_ds = utils.read_json(ds_info["ds_train"])
    num_classes = len(coco_ds["categories"])

    cnn_cfgs = {}
    for cnn in cnns:
        cnn_cfgs[cnn["name"]] = []
        base_cfg = get_cfg()

        # D2 config
        if is_key_set(cnn, "cfg_url"):
            base_cfg.merge_from_file(
                utils.join_paths_str(d2_configs_root, cnn["cfg_url"])
            )
        # my base config (e.g. base_configs/cfg_lz.yaml)
        base_cfg.merge_from_file(
            ds_info["base_cfg"]
        )

        base_cfg.DATASETS.TRAIN = (f"{ds_info['ds_name']}_train",)
        base_cfg.DATASETS.TEST = (f"{ds_info['ds_name']}_val", )
        #  D2 weights
        if is_key_set(cnn, "weight_url"):
            base_cfg.MODEL.WEIGHTS = cnn["weight_url"]
        base_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # ds pixel mean, pixel std
        px_mean, px_std = calc_pixel_mean_std(ds_info)
        base_cfg.MODEL.PIXEL_MEAN = px_mean
        base_cfg.MODEL.PIXEL_STD = px_std

        # get cartesian product of all parameters
        param_permuts = list(cart_product_dict(**parameters))

        for perm in param_permuts:
            # create configs
            cfg = copy.deepcopy(base_cfg)

            out_dir_name = ""
            for d2_key, val in perm.items():
                set_d2_cfg_attr(cfg, {"detectron_field": d2_key, "value": val})
                parts = d2_key.split(".")
                field = parts[1] if len(parts) > 1 else parts[0]
                out_dir_name += f"{field}_{val}" if out_dir_name == "" else f"_{field}_{val}"

            out_path = utils.join_paths_str(ds_info["ds_path"], OUT_DIR_ROOT, cnn["name"], out_dir_name)

            cfg.OUTPUT_DIR = out_path
            cnn_cfgs[cnn["name"]].append(cfg)
    return cnn_cfgs


def get_ds_info(ds_path, base_cfg=None):
    ds_name = utils.get_nth_parentdir(ds_path)
    dataset_info = {}
    dataset_info["ds_name"] = ds_name
    dataset_info["ds_path"] = ds_path
    dataset_info["ds_full"] = utils.join_paths_str(ds_path, COCO)
    dataset_info["ds_train"] = utils.join_paths_str(ds_path, COCO_TRAIN)
    dataset_info["ds_val"] = utils.join_paths_str(ds_path, COCO_VAL)
    dataset_info["ds_test"] = utils.join_paths_str(ds_path, COCO_TEST)
    if base_cfg:
        dataset_info["base_cfg"] = base_cfg
    dataset_info["image_path"] = utils.prompt_folder_confirm(dataset_info["ds_path"], IN_IMG_FOLDERS, 'images')
    return dataset_info
