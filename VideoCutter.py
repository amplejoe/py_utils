#!/usr/bin/env python

# +++++++++++++++++++++++++++++++++++++++++++++++++
#                         _       _
#    __ _ _ __ ___  _ __ | | ___ (_) ___   ___
#   / _` | '_ ` _ \| '_ \| |/ _ \| |/ _ \ / _ \
#  | (_| | | | | | | |_) | |  __/| | (_) |  __/
#   \__,_|_| |_| |_| .__/|_|\____/ |\___/ \___|
#                  |_|         |__/
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

# @Author: Andreas <aleibets>
# @Date: 2020-01-27T11:53:43+01:00
# @Filename: main_template.py
# @Last modified by: aleibets
# @Last modified time: 2020-02-12T16:07:40+01:00
# @description: cuts videos from position to position in time, requires 'ffmpeg'

from . import utils, ecat_tools
import datetime
import time
from tqdm import tqdm

# IN_EXTENSIONS = [".mp4", ".avi"]
TIME_FORMAT = "%H:%M:%S"

# constant rate factor (crf) 0-51 ->
# 0 is lossless but creates files bigger than the original, 17 or 18 to be visually lossless or nearly so
# https://trac.ffmpeg.org/wiki/Encode/H.264#:~:text=1.,sane%20range%20is%2017%E2%80%9328.
CONVERSIONS = {
    "default": "-c copy",
    "x264": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -c:a aac",
    "x264_na": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -an",
    "test": "-crf 20 -b:v 1M -maxrate 1M",
}

# from/to/duration format -> hh:mm:ss
class VideoCutter:
    def __init__(self, in_video, **kwargs):
        """Cuts videos from position to position in time, requires 'ffmpeg'

        Args:
            in_video (str):     path to video

        kwargs:
            conversions (str):  (optional) applied conversions (see CONVERSIONS)
            out_root (str):     (optional) path to output directory (default: path of source video)
        """
        self.video = in_video
        self.fps = ecat_tools.get_fps(self.video)
        self.total_duration = ecat_tools.get_duration(self.video)

        self.out_root = None
        self.conversion_string = CONVERSIONS["default"]

        # sanity checks
        if not utils.exists_file(self.video):
            exit(f"Input video not found: {self.video}")

        if "conversions" in kwargs.keys():
            if kwargs["conversions"] not in CONVERSIONS.keys():
                print("Invalid conversion key given, using default...")
            else:
                self.conversion_string = CONVERSIONS[kwargs["conversions"]]

        # out dir
        in_root = utils.get_file_path(self.video)
        self.out_root = in_root
        if "out_root" in kwargs.keys():
            self.out_root = kwargs["out_root"]
        if not utils.exists_dir(self.out_root):
            utils.make_dir(self.out_root)

        # for video re-encoding
        self.tmp_file_path = utils.join_paths(
            self.out_root,
            f"TMP_{utils.get_file_name(self.video)}_TMP{utils.get_file_ext(self.video)}",
        )

    # def seconds_to_str(self, secs):
    #     return str(datetime.timedelta(seconds=secs))

    def str_to_datetime(self, hh_mm_ss: str) -> datetime:
        """hh:mm:ss -> datetime"""
        res = None
        try:
            res = datetime.datetime.strptime(hh_mm_ss, TIME_FORMAT)
        except ValueError:
            exit(f"Input format incorrect: {hh_mm_ss}. Times given should be hh:mm:ss.")
        return res

    def datetime_to_str(self, datetime_object: datetime) -> str:
        """datetime -> hh:mm:ss"""
        return datetime_object.strftime(TIME_FORMAT)

    def secs_to_time_format(self, secs: int) -> str:
        return time.strftime(TIME_FORMAT, time.gmtime(secs))

    def time_format_to_secs(self, timestring) -> int:
        pt = datetime.datetime.strptime(timestring, TIME_FORMAT)
        total_seconds = pt.second + pt.minute * 60 + pt.hour * 3600
        return total_seconds

    def calc_duration(self, start_time: str, end_time: str) -> datetime:
        """(start_hh:mm:ss, end_hh:mm:ss) -> <datetime> duration"""
        startDateTime = self.str_to_datetime(start_time)
        endDateTime = self.str_to_datetime(end_time)
        return self.str_to_datetime(f"{endDateTime - startDateTime}")

    def calc_to_time(self, start_time: str, duration_time: str) -> datetime:
        """(start_hh:mm:ss, duration_hh:mm:ss) -> <datetime> end"""
        startDateTime = self.str_to_datetime(start_time)
        parts = duration_time.split(":")
        duration_time_hours = int(parts[0])
        duration_time_minutes = int(parts[1])
        duration_time_seconds = int(parts[2])
        to_add = datetime.timedelta(
            hours=duration_time_hours,
            minutes=duration_time_minutes,
            seconds=duration_time_seconds,
        )
        added = startDateTime + to_add
        return added

    def re_encode_with_keyframes(self, kf_list):
        string_list = ",".join(kf_list)
        cmd = f"ffmpeg -i {self.video} -force_key_frames {string_list} {self.conversion_string} {self.tmp_file_path}"
        utils.exec_shell_command(cmd, silent=True)

    def cut(self, **kwargs):
        """Single cut video with parameters

        kwargs:
        from_time (str):    (optional) hh:mm:ss (default -> 00:00:00)
        to_time (str):      (optional - alt: frames/duration) hh:mm:ss (default if no alts given: video length)
        duration (str):     (optional - alt: to_time/frames) hh:mm:ss (default if no alts given: video length)
        frames (str):       (optional - alt: to_time/duration) number of frames (default if no alts given: video length)
        out_file (str):     (optional) output file name (default: input file name with timestamps)
        """

        # sanity checks
        if len(kwargs.keys() & {"to_time", "duration", "frames"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "to_time" or "duration" (hh:mm:ss) or "frames" (int)'
            )

        # from time
        ft_input = "00:00:00"
        if "from_time" in kwargs.keys():
            ft_input = kwargs["from_time"]
        ft = self.str_to_datetime(ft_input)
        from_time = {
            "str": self.datetime_to_str(ft),
            "time": self.str_to_datetime(ft_input),
        }

        # CALC to_time (tt) and duration_time (dt)
        # default: full duration from given from_time
        dt = self.str_to_datetime(self.total_duration)
        tt = self.calc_to_time(from_time["str"], self.total_duration)
        if "to_time" in kwargs.keys():
            tt = self.str_to_datetime(kwargs["to_time"])
            # FFMPEG works with duration, need to calculate start-end difference
            dt = self.calc_duration(from_time["str"], kwargs["to_time"])
        elif "duration" in kwargs.keys():
            dt = self.str_to_datetime(kwargs["duration"])
            tt = self.calc_to_time(from_time["str"], kwargs["duration"])
        elif "frames" in kwargs.keys():
            secs = int(kwargs["frames"]) / self.fps
            d_input = self.secs_to_time_format(secs)
            dt = self.str_to_datetime(d_input)
            tt = self.calc_to_time(from_time["str"], d_input)
            # print(f"{utils.get_file_name(self.video)}: f {kwargs['frames']} fps {self.fps} -> {secs}s")

        duration = {"str": self.datetime_to_str(dt), "time": dt}
        to_time = {"str": self.datetime_to_str(tt), "time": tt}

        # build output file: timestamped of via given parameter
        in_file_name = utils.get_file_name(self.video)
        in_file_ext = utils.get_file_ext(self.video)
        out_file = f"{in_file_name}_({from_time['str'].replace(':', '-')})-({to_time['str'].replace(':', '-')}){in_file_ext}"
        if "out_file" in kwargs.keys():
            out_file = kwargs["out_file"]

        # full out path
        out_path = utils.join_paths(
            self.out_root,
            out_file,
        )

        # DEBUG
        # print(f"from\n\t{from_time}")
        # print(f"to\n\t{to_time}")
        # print(f"dur\n\t{duration}")
        # print(f"dur\n\t{self.total_duration}")
        # print(f"out {out_file}")

        # 2021: performance improvement -> directly re-encode with cut command
        # re-encode setting from and to as keyframes
        # self.re_encode_with_keyframes([from_time["str"], to_time["str"]])
        # keyframe_list = [from_time["str"], to_time["str"]]
        # keyframe_list_str = ",".join(keyframe_list)

        # 'to' is the duration of the cutout
        cmd = f"ffmpeg -ss {from_time['str']} -i {self.video} -to {duration['str']} {self.conversion_string} {out_path}"

        # HACK: frame extraction with duration is not accurate -> use filter
        if "frames" in kwargs.keys():
            from_time_secs = self.time_format_to_secs(ft_input)
            from_frame = int(from_time_secs * self.fps)
            to_frame = int(kwargs['frames']) - 1 # frames start at 0
            cmd = f'ffmpeg -i {self.video} -vf select="between(n\,{from_frame}\,{to_frame}),setpts=PTS-STARTPTS" {self.conversion_string} {out_path}'

        utils.exec_shell_command(cmd, silent=True)

        # clean up tmp file
        # utils.remove_file(self.tmp_file_path)

    def multi_cut(self, **kwargs):
        """Multi cut video with parameters, i.e. every S seconds or every F frames.

        kwargs:
        seconds (int):      (optional) periodic cut interval in seconds
        frames (int):       (optional - alt: to_time/duration) number of frames (default if no alts given: video length)
        out_file (str):     (optional) output file name (default: input file name with timestamps)
        """

        # sanity checks
        if len(kwargs.keys() & {"seconds", "frames"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "seconds" (int) or "frames" (int)'
            )

        # variables
        in_file_name = utils.get_file_name(self.video)
        in_file_ext = utils.get_file_ext(self.video)
        num_seconds = None
        if "seconds" in kwargs.keys():
            num_seconds = kwargs["seconds"]
        elif "frames" in kwargs.keys():
            num_seconds = int(kwargs["frames"]) / self.fps

        # re-encode video making keyframes in given seconds interval
        total_seconds = self.time_format_to_secs(self.total_duration)
        kf_list = []
        for cur_sec in range(0, total_seconds, num_seconds):
            from_kf_sec = cur_sec
            to_kf_sec = cur_sec + num_seconds
            from_kf_str = self.secs_to_time_format(from_kf_sec)
            to_kf_str = self.secs_to_time_format(to_kf_sec)
            if to_kf_sec > total_seconds:
                to_kf_str = self.secs_to_time_format(total_seconds)
            kf_list.append(from_kf_str)
            kf_list.append(to_kf_str)
        # init progress bar (before video encoding for prettier output)
        kf_list_pbar = tqdm(
            range(0, len(kf_list), 2),
            desc=f"{in_file_name}{in_file_ext}",
            position=0,
            leave=False,
        )
        self.re_encode_with_keyframes(kf_list)

        # cut video in intervals
        for i in kf_list_pbar:
            from_time_str = kf_list[i]
            from_time_secs = self.time_format_to_secs(from_time_str)
            to_time_str = kf_list[i + 1]
            duration_str = self.datetime_to_str(
                self.calc_duration(from_time_str, to_time_str)
            )
            out_file = f"{in_file_name}_{from_time_secs}{in_file_ext}"
            out_path = utils.join_paths(self.out_root, out_file)
            # '-to' denotes the duration
            cmd = f"ffmpeg -ss {from_time_str} -i {self.tmp_file_path} -to {duration_str} -c copy {out_path}"
            utils.exec_shell_command(cmd, silent=True)

        # clean up tmp file
        utils.remove_file(self.tmp_file_path)
