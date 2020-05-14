from tqdm import tqdm
from . import utils
import copy
import itertools
import functools

# detectron2 imports
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

SETTINGS_FILE = "settings.json"
IN_IMG_FOLDERS = ['img', 'imag', 'frame', 'pic', 'phot']

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


def create_d2_cfgs(ds_settings, d2_configs_root):
    """ Get detectron2 configs for a specific dataset. Returns list of cfgs depending on parameters
    """
    img_folder = utils.prompt_folder_confirm(ds_settings['ds_path'], IN_IMG_FOLDERS, 'images')

    coco_ds = utils.read_json(ds_settings["ds_train"])
    num_classes = len(coco_ds["categories"])

    # register datasets
    register_coco_instances(f"{ds_settings['ds_name']}_train", {}, ds_settings["ds_train"], img_folder)
    register_coco_instances(f"{ds_settings['ds_name']}_val", {}, ds_settings["ds_val"], img_folder)
    register_coco_instances(f"{ds_settings['ds_name']}_test", {}, ds_settings["ds_test"], img_folder)

    script_dir = utils.get_script_dir()
    global_settings_path = utils.join_paths_str(script_dir, GLOBAL_SETTINGS)
    global_settings = utils.read_json(global_settings_path)

    cnn_cfgs = {}
    for cnn in ds_settings["cnns"]:
        cnn_cfgs[cnn["name"]] = []
        base_cfg = get_cfg()

        if is_key_set(cnn, "cfg_url"):
            base_cfg.merge_from_file(
                utils.join_paths_str(d2_configs_root, cnn["cfg_url"])
            )

        base_cfg.DATASETS.TRAIN = (f"{ds_settings['ds_name']}_train",)
        base_cfg.DATASETS.TEST = (f"{ds_settings['ds_name']}_val", )
        if is_key_set(cnn, "weight_url"):
            base_cfg.MODEL.WEIGHTS = cnn["weight_url"]
        base_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        for setting in global_settings:
            set_d2_cfg_attr(base_cfg, setting, is_key_enabled(setting, "eval"))

        params_dict = params_list_to_dict(ds_settings["parameters"])

        # get cartesian product of all parameters
        param_permuts = list(cart_product_dict(**params_dict))

        for perm in param_permuts:
            # create configs
            cfg = copy.deepcopy(base_cfg)

            out_dir_name = ""
            for d2_key, val in perm.items():
                set_d2_cfg_attr(cfg, {"detectron_field": d2_key, "value": val})
                parts = d2_key.split(".")
                field = parts[1] if len(parts) > 1 else parts[0]
                out_dir_name += f"{field}_{val}" if out_dir_name == "" else f"_{field}_{val}"

            out_path = utils.join_paths_str(ds_settings["ds_path"], OUT_DIR_ROOT, out_dir_name)

            cfg.OUTPUT_DIR = out_path
            cnn_cfgs[cnn["name"]].append(cfg)
    return cnn_cfgs


def get_ds_settings(ds_path):
    ds_name = utils.get_nth_parentdir(ds_path)
    dataset_info = {}
    dataset_info["ds_name"] = ds_name
    dataset_info["ds_path"] = ds_path
    dataset_info["ds_full"] = utils.join_paths_str(ds_path, COCO)
    dataset_info["ds_train"] = utils.join_paths_str(ds_path, COCO_TRAIN)
    dataset_info["ds_val"] = utils.join_paths_str(ds_path, COCO_VAL)
    dataset_info["ds_test"] = utils.join_paths_str(ds_path, COCO_TEST)
    settings_file = utils.join_paths_str(dataset_info["ds_path"], SETTINGS_FILE)
    ds_settings = utils.read_json(settings_file)
    ds_settings = {**dataset_info, **ds_settings}  # merge dicts
    return ds_settings
