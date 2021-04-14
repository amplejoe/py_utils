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

# IN_EXTENSIONS = [".mp4", ".avi"]
TIME_FORMAT = "%H:%M:%S"

# from/to/duration format -> hh:mm:ss
class VideoCutter:
    def __init__(self, in_video, **kwargs):
        """Cuts videos from position to position in time, requires 'ffmpeg'

        Args:
            in_video (str):     path to video

        kwargs:
            from_time (str):    (optional) hh:mm:ss (default -> 00:00:00)
            to_time (str):      (optional - alt: frames/duration) hh:mm:ss (default if no alts given: video length)
            duration (str):     (optional - alt: to_time/frames) hh:mm:ss (default if no alts given: video length)
            frames (str):       (optional - alt: to_time/duration) number of frames (default if no alts given: video length)
            out_root (str):     (optional) path to output directory (default: path of source video)
            out_file (str):     (optional) output file name (default: input file name with timestamps)
        """
        self.video = in_video
        self.fps = ecat_tools.get_fps(self.video)
        self.total_duration = ecat_tools.get_duration(self.video)
        self.from_time = None
        self.to_time = None
        self.duration = None
        self.frames = None
        self.out_root = None
        self.out_file = None

        # sanity checks
        if len(kwargs.keys() & {"to_time", "duration", "frames"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "to_time" or "duration" (hh:mm:ss) or "frames" (int)'
            )
        if not utils.exists_file(self.video):
            exit(f"Input video not found: {self.video}")

        # from time
        ft_input = "00:00:00"
        if "from_time" in kwargs.keys():
            ft_input = kwargs["from_time"]
        ft = self.str_to_datetime(ft_input)
        self.from_time = {
            "str": self.datetime_to_str(ft),
            "time": self.str_to_datetime(ft_input),
        }

        # out dir
        in_root = utils.get_file_path(self.video)
        self.out_root = in_root
        if "out_root" in kwargs.keys():
            self.out_root = kwargs["out_root"]

        # CALC to_time (tt) and duration_time (dt)
        # default: full duration from given from_time
        dt = self.str_to_datetime(self.total_duration)
        tt = self.get_to_time(self.from_time["str"], self.total_duration)
        if "to_time" in kwargs.keys():
            tt = self.str_to_datetime(kwargs["to_time"])
            # FFMPEG works with duration, need to calculate start-end difference
            dt = self.get_duration(self.from_time["str"], kwargs["to_time"])
        elif "duration" in kwargs.keys():
            dt = self.str_to_datetime(kwargs["duration"])
            tt = self.get_to_time(self.from_time["str"], kwargs["duration"])
        elif "frames" in kwargs.keys():
            secs = int(kwargs["frames"]) / self.fps
            d_input = self.secs_to_time_format(secs)
            dt = self.str_to_datetime(d_input)
            tt = self.get_to_time(self.from_time["str"], d_input)

        self.duration = {"str": self.datetime_to_str(dt), "time": dt}
        self.to_time = {"str": self.datetime_to_str(tt), "time": tt}

        # out file
        in_file_name = utils.get_file_name(self.video)
        in_file_ext = utils.get_file_ext(self.video)
        self.out_file = f"{in_file_name}_({self.from_time['str'].replace(':', '-')})-({self.to_time['str'].replace(':', '-')}){in_file_ext}"
        if "out_file" in kwargs.keys():
            self.out_file = kwargs["out_file"]

        # DEBUG
        # print(f"from\n\t{self.from_time}")
        # print(f"to\n\t{self.to_time}")
        # print(f"dur\n\t{self.duration}")
        # print(f"out {self.out_file}")

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

    def get_duration(self, start_time: str, end_time: str) -> datetime:
        """(start_hh:mm:ss, end_hh:mm:ss) -> <datetime> duration"""
        startDateTime = self.str_to_datetime(start_time)
        endDateTime = self.str_to_datetime(end_time)
        return self.str_to_datetime(f"{endDateTime - startDateTime}")

    def get_to_time(self, start_time: str, duration_time: str) -> datetime:
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

    def start(self):

        out_path = utils.join_paths(
            self.out_root,
            self.out_file,
        )

        # 'to' is the duration of the cutout
        cmd = f"ffmpeg -ss {self.from_time['str']} -i {self.video} -to {self.duration['str']} -c copy {out_path}"

        utils.exec_shell_command(cmd, silent=True)
