from tqdm import tqdm
from . import utils

DEFAULT_CFG_FILE = "config.json"
DEFAULT_IMG_DIRS = ['img', 'imag', 'frame', 'pic', 'phot']
DEFAULT_ANNOT_DIRS = ['annot', 'sketch', 'mask']

def get_ds_config(ds_root, *, has_annots=False):
    cfg_location = utils.join_paths_str(ds_root, DEFAULT_CFG_FILE)
    cfg = utils.read_json(cfg_location)
    cfg_update = {}

    if ("images" not in cfg):
        cfg_update["images"] = utils.prompt_folder_confirm(ds_root, DEFAULT_IMG_DIRS, "images")

    if has_annots:
        if ("images" not in cfg):
            ret["annotations"] = utils.join_paths(ds_root, cfg["annotations"])
            cfg_update["annotations"] = utils.prompt_folder_confirm(ds_root, DEFAULT_ANNOT_DIRS, "annotations")

    utils.update_config_file(cfg_location, cfg_update)
           
    # re-read pot. changed config
    cfg = utils.read_json(cfg_location)
    # runtime parameter (config's path)
    cfg['config_path'] = cfg_location
 
    return cfg