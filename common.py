from . import utils

DEFAULT_CFG_FILE = "config.json"
DEFAULT_IMG_DIRS = ['img', 'imag', 'frame', 'pic', 'phot']
DEFAULT_ANNOT_DIRS = ['annot', 'sketch', 'mask']

ANNOTATION_KEY = "annotations"
IMAGE_KEY = "images"

def get_ds_config(ds_root, *, has_annots=False):
    cfg_location = utils.join_paths_str(ds_root, DEFAULT_CFG_FILE)
    cfg = utils.read_json(cfg_location)
    cfg_update = {}

    if (not cfg or IMAGE_KEY not in cfg):
        full_frame_dir = utils.prompt_folder_confirm(ds_root, DEFAULT_IMG_DIRS, IMAGE_KEY)
        cfg_update[IMAGE_KEY] = utils.path_to_relative_path(full_frame_dir, ds_root)

    if has_annots:
        if (not cfg or ANNOTATION_KEY not in cfg):
            full_annot_dir = utils.prompt_folder_confirm(ds_root, DEFAULT_ANNOT_DIRS, ANNOTATION_KEY)
            cfg_update[ANNOTATION_KEY] = utils.path_to_relative_path(full_annot_dir, ds_root)
    utils.update_config_file(cfg_location, cfg_update)

    # re-read pot. changed config
    cfg = utils.read_json(cfg_location)

    # runtime parameters (never saved to file!!)
    cfg['config_path'] = cfg_location
    cfg[f'{IMAGE_KEY}_full'] = utils.join_paths(ds_root, cfg[IMAGE_KEY])
    if has_annots:
        cfg[f'{ANNOTATION_KEY}_full'] = utils.join_paths(ds_root, cfg[ANNOTATION_KEY])

    return cfg