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

from datetime import datetime, timedelta

from tqdm import tqdm

# IN_EXTENSIONS = [".mp4", ".avi"]

# 2021: more precision Hours:Minutes:Seconds:Microseconds
TIME_FORMAT = "%H:%M:%S.%f"

# constant rate factor (crf) 0-51 ->
# 0 is lossless but creates files bigger than the original, 17 or 18 to be visually lossless or nearly so
# https://trac.ffmpeg.org/wiki/Encode/H.264#:~:text=1.,sane%20range%20is%2017%E2%80%9328.
CONVERSIONS = {
    "default": "-c copy",
    "x264": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -c:a aac",
    "x264_na": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -an",
    "test": "-crf 20 -b:v 1M -maxrate 1M",
}


class VideoCutter:
    def __init__(self, in_video, **kwargs):
        """Cuts videos from position to position in time, requires 'ffmpeg'

        Args:
            in_video (str):     path to video

        kwargs:
            conversions (str):  (optional) applied conversions (see CONVERSIONS)
            out_root (str):     (optional) path to output directory (default: path of source video)
            force_overwrite (bool): (optional) sets overwrite flags for ffmpeg (default: False)
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

        self.overwrite_flag = "-n"
        if "force_overwrite" in kwargs.keys():
            self.overwrite_flag = "-y" if kwargs["force_overwrite"] else "-n"

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

    def str_to_timedelta(self, h_m_s_f: str) -> timedelta:
        # we specify the input and the format...
        t = datetime.strptime(h_m_s_f, TIME_FORMAT)
        # ...and use datetime's hour, min and sec properties to build a timedelta
        return timedelta(
            hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond
        )

    def timedelta_to_str(self, datetime_object: timedelta) -> str:
        return str(datetime_object)

    def secs_to_timedelta(self, sec_float: float) -> str:
        seconds = int(sec_float)
        microseconds = (sec_float * 1000000) % 1000000
        return timedelta(0, seconds, microseconds)

    def timedelta_to_secs(self, td: timedelta) -> float:
        sec = td.seconds
        mic = td.microseconds
        sec_float = sec + (mic / 1000000.0)
        return sec_float

    def frame_to_secs(self, frame):
        return int(frame) / self.fps

    def secs_to_frame(self, time_in_secs):
        return int(time_in_secs * self.fps)

    # def calc_duration(self, start_time: str, end_time: str) -> datetime:
    #     """(start_hh:mm:ss.micro, end_hh:mm:ss.micro) -> <datetime> duration"""
    #     startDateTime = self.str_to_datetime(start_time)
    #     endDateTime = self.str_to_datetime(end_time)
    #     return self.str_to_datetime(f"{endDateTime - startDateTime}")

    # def calc_to_time(self, start_time: str, duration_time: str) -> datetime:
    #     """(start_hh:mm:ss.micro, duration_hh:mm:ss.micro) -> <datetime> end"""
    #     startDateTime = self.str_to_datetime(start_time)
    #     parts = duration_time.split(":")
    #     duration_time_hours = int(parts[0])
    #     duration_time_minutes = int(parts[1])
    #     duration_time_seconds = int(parts[2])
    #     to_add = datetime.timedelta(
    #         hours=duration_time_hours,
    #         minutes=duration_time_minutes,
    #         seconds=duration_time_seconds,
    #     )
    #     added = startDateTime + to_add
    #     return added

    def re_encode_with_keyframes(self, kf_list):
        string_list = ",".join(kf_list)
        cmd = f"ffmpeg {self.overwrite_flag} -i {self.video} -force_key_frames {string_list} {self.conversion_string} {self.tmp_file_path}"
        utils.exec_shell_command(cmd, silent=True)

    def cut(self, **kwargs):
        """Single cut video with parameters

        kwargs:
        from_time (str):    (optional) hh:mm:ss.micro (default -> 00:00:00.0)
        from_frame (str):   (optional - alt: from_time)
        to_time (str):      (optional - alt: frames/duration) hh:mm:ss.micro (default if no alts given: video length)
        duration (str):     (optional - alt: to_time/frames) hh:mm:ss.micro (default if no alts given: video length)
        frames (str):       (optional - alt: to_time/duration) number of frames (default if no alts given: video length)
        out_file (str):     (optional) output file name (default: input file name with timestamps)
        """

        # sanity checks
        if len(kwargs.keys() & {"to_time", "duration", "frames"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "to_time" or "duration" (hh:mm:ss.micro) or "frames" (int)'
            )
        if len(kwargs.keys() & {"from_time", "from_frame"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "from_time" (hh:mm:ss.micro) or "from_frame"'
            )

        # from time
        ft_input = "00:00:00.0"
        if "from_time" in kwargs.keys():
            ft_input = kwargs["from_time"]
        elif "from_frame" in kwargs.keys():
            in_secs = self.frame_to_secs(kwargs["from_frame"])
            ft_input = str(self.secs_to_timedelta(in_secs))
        from_time = self.str_to_timedelta(ft_input)

        # CALC to_time (tt) and duration_time (dt)
        # default: full duration from given from_time
        duration_time = self.str_to_timedelta(self.total_duration)
        to_time = from_time + duration_time
        if "to_time" in kwargs.keys():
            to_time = self.str_to_timedelta(kwargs["to_time"])
            # FFMPEG works with duration, need to calculate start-end difference
            duration_time = to_time - from_time
        elif "duration" in kwargs.keys():
            duration_time = self.str_to_timedelta(kwargs["duration"])
            to_time = from_time + duration_time
        elif "frames" in kwargs.keys():
            secs_float = int(kwargs["frames"]) / self.fps
            duration_time = self.secs_to_timedelta(secs_float)
            to_time = from_time + duration_time
            # print(f"{utils.get_file_name(self.video)}: f {kwargs['frames']} fps {self.fps} -> {secs}s")

        # build output file: timestamped of via given parameter
        in_file_name = utils.get_file_name(self.video)
        in_file_ext = utils.get_file_ext(self.video)
        out_file = f"{in_file_name}_({str(from_time).replace(':', '-')})-({str(to_time).replace(':', '-')}){in_file_ext}"
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
        # print(f"fps\n\t{self.fps}")
        # print(f"dur\n\t{duration}")
        # print(f"dur\n\t{self.total_duration}")
        # print(f"out {out_file}")

        # 2021: performance improvement -> directly re-encode with cut command
        # re-encode setting from and to as keyframes
        # self.re_encode_with_keyframes([from_time["str"], to_time["str"]])
        # keyframe_list = [from_time["str"], to_time["str"]]
        # keyframe_list_str = ",".join(keyframe_list)

        # 'to' is the duration of the cutout
        cmd = f'ffmpeg {self.overwrite_flag} -ss {str(from_time)} -i {self.video} -to {str(duration_time)} {self.conversion_string} "{out_path}"'

        # HACK: frame extraction with duration is not accurate -> use filter
        if "frames" in kwargs.keys():
            from_time_secs = self.timedelta_to_secs(from_time)
            from_frame = self.secs_to_frame(from_time_secs)
            to_frame = from_frame + (
                int(kwargs["frames"]) - 1
            )  # last frame is inclusive subtract 1 to get exact #frames
            cmd = f'ffmpeg {self.overwrite_flag} -i {self.video} -an -vf select="between(n\,{from_frame}\,{to_frame}),setpts=PTS-STARTPTS" {self.conversion_string} "{out_path}"'

        utils.exec_shell_command(cmd, silent=True)

        # clean up tmp file
        # utils.remove_file(self.tmp_file_path)

    def multi_cut(self, **kwargs):
        """Multi cut video with parameters, i.e. every S seconds or every F frames.

        kwargs:
        seconds (float):    (optional) periodic cut interval in seconds
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
            num_seconds = float(kwargs["seconds"])
        elif "frames" in kwargs.keys():
            num_seconds = int(kwargs["frames"]) / self.fps

        # re-encode video making keyframes in given seconds interval
        duration = self.str_to_timedelta(self.total_duration)
        total_seconds = self.timedelta_to_secs(duration)
        kf_list = []
        for cur_sec in range(0, total_seconds, num_seconds):
            from_kf_sec = cur_sec
            to_kf_sec = cur_sec + num_seconds
            from_kf = self.secs_to_timedelta(from_kf_sec)
            to_kf = self.secs_to_timedelta(to_kf_sec)
            if to_kf_sec > total_seconds:
                to_kf = self.secs_to_timedelta(total_seconds)
            kf_list.append(from_kf)
            kf_list.append(to_kf)
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
            from_time = kf_list[i]
            from_time_secs = self.timedelta_to_secs(from_time)
            to_time = kf_list[i + 1]
            duration = to_time - from_time
            out_file = f"{in_file_name}_{from_time_secs}{in_file_ext}"
            out_path = utils.join_paths(self.out_root, out_file)
            # '-to' denotes the duration
            cmd = f"ffmpeg {self.overwrite_flag} -ss {str(from_time)} -i {self.tmp_file_path} -to {str(duration)} -c copy '{out_path}'"
            utils.exec_shell_command(cmd, silent=True)

        # clean up tmp file
        utils.remove_file(self.tmp_file_path)
