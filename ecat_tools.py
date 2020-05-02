#!/usr/bin/env python
import json
import pandas as pd
from tqdm import tqdm
import json
import random
from . import utils

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
    """ Extracts video-case mapping info from ECAT mapping files
        Returns : dict{video_id: case_id}, dict{case_id: video_count}
    """
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


def get_info(frame_or_sketch_or_vid):
    if "video_" not in frame_or_sketch_or_vid:
        # invalid file path
        return None

    ret_dict = {}
    ret_dict["path"] = frame_or_sketch_or_vid
    ret_dict["file_name"] = utils.get_file_name(frame_or_sketch_or_vid)
    ret_dict["file_ext"] = utils.get_file_ext(frame_or_sketch_or_vid)

    # find last occurrence of '_video'
    tmp = frame_or_sketch_or_vid.rsplit("video_")[1].replace(".mp4", "")
    tmp_parts = tmp.split("/")[0].split("_")  # remove frame part if existent
    ret_dict["video_id"] = tmp_parts[0]
    ret_dict["start_time"] = float(tmp_parts[1])
    ret_dict["end_time"] = ret_dict["start_time"]
    if len(tmp_parts) > 2:
        ret_dict["end_time"] = float(tmp_parts[2])

    if ret_dict["file_ext"] == ".jpg":
        ret_dict["frame"] = int(ret_dict["file_name"].split("_")[1])
    elif ret_dict["file_ext"] == ".json":
        ret_dict["frame"] = get_sketch_frame(ret_dict["path"])
    else:
        ret_dict["fps"] = get_fps(ret_dict["path"])
        ret_dict["start_frame"] = time_to_frame(ret_dict["start_time"], ret_dict["fps"])
        ret_dict["end_frame"] = time_to_frame(ret_dict["end_time"], ret_dict["fps"])
    return ret_dict


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


def get_fps_from_sketch(sketch_file_path):
    """ gets video fps from sketch file
    """
    fps = -1
    # get fps
    with open(sketch_file_path) as json_file:
        data = json.load(json_file)
        fps = float(data['fps'])
        return fps


def get_sketch_timecode(sketch_file_name):
    """ gets timestamp from sketch file name
        sketch_file_name format: sg_SGID_a_AID_s_SID_t_TSTAMP.json
        with segment sg, annotation a, sketch s and timestamp t
    """
    return float(sketch_file_name.split("t_")[1])


def get_sketch_frame(sketch_file_path):
    """ calculates a sketch's frame number via its json file path
    """
    file_name = utils.get_file_name(sketch_file_path)
    fps = get_fps_from_sketch(sketch_file_path)
    time = get_sketch_timecode(file_name)
    return time_to_frame(time, fps)


def unescape(in_string):
    """ unescapes strings, used for SVG paths like:
        <rect x=\"-146.56\" y=\"-85.12\" rx=\"0\" ry=\"0\" width=\"293.12\" height=\"170.24\" style=\"stroke: rgb(255,255,255); stroke-width: 2; stroke-dasharray: none; stroke-linecap: butt; stroke-linejoin: miter; stroke-miterlimit: 4; fill: rgb(0,0,0); fill-opacity: 0; fill-rule: nonzero; opacity: 1;\" transform=\"translate(236.73 156.52)\"\/>\n
    """
    return bytes(in_string, encoding='utf8').decode('unicode_escape').replace("\/", "/")


def prepare_svg(svg_tag, color_rgb):
    """ Prepare svg for writing

        Parameters
        ----------

        svg_tag : str
        svg to be written to disk
        color_rgb : str
        rgb color in CSS format: 'rgb(r,g,b)'
    """
    #  fill sketches
    svg_tag = unescape(svg_tag)

    # color_str = f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})"
    svg_tag = svg_tag.replace("fill: rgb(0,0,0);", f"fill: {color_rgb};")
    svg_tag = svg_tag.replace("stroke: rgb(255,255,255);", f"stroke: {color_rgb};")
    svg_tag = svg_tag.replace("fill-opacity: 0;", "fill-opacity: 1;")
    return svg_tag


def create_svg(svg_tag, img_width, img_height, out_path):
    """ creates an svg from an xml tag describing a drawing (ECAT export)
    """
    with open("template.svg", "rt") as fin:
        with open(out_path, "wt") as fout:
            for line in fin:
                fout.write(line.replace("INSERT_WIDTH", str(img_width))
                           .replace("INSERT_HEIGHT", str(img_height))
                           .replace("INSERT_OBJECT", svg_tag))


# https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
# does not generate very distinct colors, use random_colors.py instead
# def get_distinct_colors(num_colors):
#     """ Generate num_colors random distinct RGB colors
#     """
#     ret = []
#     # don't generate black (0,0,0) or white (255,255,255)
#     # ORIGINAL: int(random.random() * 256)
#     r = random.randint(1, 255)
#     g = random.randint(1, 255)
#     b = random.randint(1, 255)
#     step = 256 / num_colors
#     for i in range(num_colors):
#         r += step
#         g += step
#         b += step
#         r = int(r) % 256
#         g = int(g) % 256
#         b = int(b) % 256
#         ret.append((r, g, b))
#     return ret


def get_sketch_frame_mapping(frame_list, sketch_list, classes_list, frames_root):
    """ Creates a sketch-frame mapping for ECAT annotation output,
        i.e. segment frames with corresponding json annotations (1 file per annotation)
        Return : dict{
                    frame_path(str): dict{
                        path:       str
                        file_name:  str
                        file_ext:   str
                        video_id:   str
                        start_time: float
                        end_time:   float
                        frame:      int
                    }
                 }
        {video_id_frame_id -> [video_frame_sketch_info_1, video_frame_sketch_info_2, ...]}
    """
    dict_frame_sketches = {}
    # for every class
    for cl in tqdm(classes_list, desc="creating sketch-frame mapping"):
        cl_frame_list = utils.filter_list_by_partial_word(cl, frame_list)
        cl_sketch_list = utils.filter_list_by_partial_word(cl, sketch_list)

        # 1. iterate over sketches (because they are fewer than frames)
        for s in cl_sketch_list:
            sketch_info = get_info(s)
            sketch_info["class"] = cl
            # 2. look for corrsponding frame
            video_frames = utils.filter_list_by_partial_word(sketch_info["video_id"], cl_frame_list)
            frame = utils.filter_list_by_partial_word(f"frame_{sketch_info['frame']}", video_frames)
            # sanity checks
            if len(frame) == 0:
                print(f"{len(frame)} frames found for sketch {sketch_info['path']}")
                continue  # prevent crashing, if no frame is found
            # frame can infact be > 1 since extracted frames/segments can oberlap
            # ignore this since frames extracted multiple times ARE still the same
            # elif len(frame) > 1:
            #   pass

            frame = frame[0]
            sketch_info["frame_path"] = frame
            # only using vid_fid as dict key ensures sketches on the same frame,
            # but of different classes to map to the same frame
            dict_key = f"v{sketch_info['video_id']}_f{sketch_info['frame']}/"
            # 3. Add sketch info to dict
            utils.add_to_dict_key(dict_frame_sketches, dict_key, sketch_info)

    return dict_frame_sketches
