#!/usr/bin/env python
import json
from py_utils import utils
import pandas as pd

SHELL_CMD_GET_FPS = "ffprobe -v 0 -of csv=p=0 -select_streams 0 -show_entries stream=r_frame_rate"


def time_to_frame(time, fps):
    return round(time * fps)


def get_video_id(vid_folder_string):
    """ gets video id from video folder name
        vid_folder_string format: video_ID_TCFROM_TCTO
    """
    parts = vid_folder_string.split("_")
    return parts[0] + "_" + parts[1]


def get_time_range(vid_folder_string):
    """ gets timerange from video folder name
        vid_folder_string format: video_ID_TCFROM_TCTO
    """
    parts = vid_folder_string.split("_")
    tc_start = -1.0
    tc_end = -1.0
    if len(parts) == 3:
        # segment is single frame
        tc_start = parts[2]
        tc_end = parts[2]
        pass
    elif len(parts) == 4:
        # segment is multiframe
        tc_start = parts[2]
        tc_end = parts[3]
    else:
        print("Invalid Segment: " + vid_folder_string)
    return float(tc_start), float(tc_end)


def get_mapping_info(in_path):
    if not utils.exists_file(in_path):
        utils.exit(f"Mapping file does not exist:\n{in_path}")
    df = pd.read_csv(in_path, sep=';', header=None)
    mapping = {}
    case_video_count = {}
    for index, row in df.iterrows():
        parts = row[0].split('_')
        video_id = "-1"
        if len(parts) == 1 or "video" not in parts[0]:
            video_id = row[0]
        else:
            video_id = parts[1]
        video_id = video_id.replace('.mp4', '')
        case_id = int(row[2])
        mapping[f'{video_id}'] = case_id
        utils.increment_dict_key(case_video_count, f'{case_id}')
    return mapping, case_video_count


def get_info(frame_or_vid):
    if "video_" not in frame_or_vid:
        # invalid file path
        return None

    ret_dict = {}
    ret_dict["path"] = frame_or_vid
    ret_dict["file_name"] = utils.get_file_name(frame_or_vid)
    ret_dict["file_ext"] = utils.get_file_ext(frame_or_vid)

    # find last aoccurrence of '_video'
    tmp = frame_or_vid.rsplit("video_")[1].replace(".mp4_", "")
    tmp_parts = tmp.split("/")[0].split("_")  # remove frame part if existent
    ret_dict["start_time"] = float(tmp_parts[0])
    ret_dict["end_time"] = ret_dict["start_time"]
    if len(tmp_parts) > 1:
        ret_dict["end_time"] = float(tmp_parts[1])

    if ret_dict["file_ext"] == "jpg":
        ret_dict["frame"] = float(ret_dict["file_name"].split("_")[1])
    else:
        ret_dict["fps"] = get_fps(ret_dict["path"])
        ret_dict["start_frame"] = time_to_frame(ret_dict["start_time"], ret_dict["fps"])
        ret_dict["end_frame"] = time_to_frame(ret_dict["end_time"], ret_dict["fps"])
    return ret_dict


def get_sketch_timecode(sketch_file_name):
    """ gets timestamp from sketch file name
        sketch_file_name format: sg_SGID_a_AID_s_SID_t_TSTAMP.json
        with segment sg, annotation a, sketch s and timestamp t
    """
    return float(sketch_file_name.split("t_")[1])


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def get_fps(video):
    """ Gets fps from a video using ffprobe
    """
    return convert_to_float(utils.exec_shell_command(SHELL_CMD_GET_FPS + " " + video)[0])
