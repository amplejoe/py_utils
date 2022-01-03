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

import cv2

from datetime import datetime, timedelta

from tqdm import tqdm

# IN_EXTENSIONS = [".mp4", ".avi"]

# 2021: more precision Hours:Minutes:Seconds:Microseconds
TIME_FORMATS = ["%H:%M:%S.%f", "%H:%M:%S"]

MODES = ["ffmpeg", "opencv"]
default_mode = MODES[0]

CONVERSIONS = {}
CONVERSIONS[MODES[0]] = {
    # constant rate factor (crf) 0-51 ->
    # 0 is lossless but creates files bigger than the original, 17 or 18 to be visually lossless or nearly so
    # https://trac.ffmpeg.org/wiki/Encode/H.264#:~:text=1.,sane%20range%20is%2017%E2%80%9328.
    "default": "-c copy",
    "x264": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -c:a aac",
    "x264_na": "-c:v libx264 -crf 20 -b:v 1M -maxrate 1M -an",
    "test": "-crf 20 -b:v 1M -maxrate 1M",
}
# check available codecs first, eg. using 'ffmpeg -codecs'
CONVERSIONS[MODES[1]] = {
    "default": {"code": "mp4v", "container": "mp4"},
    # Windows - check for appropriate releases: https://github.com/cisco/openh264/releases to place into path
    "x264": {"code": "x264", "container": "mkv"},
    "h264": {"code": "h264", "container": "mkv"},
    "avc1": {"code": "avc1", "container": "mp4"},
    "pim1": {"code": "pim1", "container": "avi"},
    "mjpg": {"code": "mjpg", "container": "mp4"},
    "vp80": {"code": "avc1", "container": "webm"},
}


class VideoCutter:
    def __init__(self, in_video, **kwargs):
        """Cuts videos from position to position in time, requires 'ffmpeg'

        Args:
            in_video (str):     path to video

        kwargs:
            mode (str):         cut mode ("ffmpeg", "opencv")
            conversions (str):  (optional) applied conversions, depends on mode (see CONVERSIONS)
            out_root (str):     (optional) path to output directory (default: path of source video)
            override_fps (str): (optional) override default fps to compensate for pot. inaccurate frame numbers (e.g. from OVAT annots)
            force_overwrite (bool): (optional) sets overwrite flags for ffmpeg (default: False)
        """
        self.video = in_video
        self.fps = ecat_tools.get_fps(self.video, use_opencv=True)
        self.override_fps = None
        if "override_fps" in kwargs.keys():
            self.override_fps = kwargs["override_fps"]
        self.total_duration = ecat_tools.get_duration(self.video)

        self.out_root = None

        self.mode = default_mode

        # sanity checks
        if not utils.exists_file(self.video):
            exit(f"Input video not found: {self.video}")

        if "mode" in kwargs.keys():
            if kwargs["mode"] not in MODES:
                print(
                    f"Invalid mode key '{kwargs['mode']}', using default '{default_mode}'"
                )
            else:
                self.mode = kwargs["mode"]

        self.conversion = CONVERSIONS[self.mode]["default"]

        if "conversions" in kwargs.keys():
            if kwargs["conversions"] not in CONVERSIONS[self.mode].keys():
                print(
                    f"Invalid conversion key '{kwargs['conversions']}', using default '{self.conversion}'"
                )
            else:
                self.conversion = CONVERSIONS[self.mode][kwargs["conversions"]]

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
        for fmt in TIME_FORMATS:
            try:
                # we specify the input and the format...
                t = datetime.strptime(h_m_s_f, fmt)
                # ...and use datetime's hour, min and sec properties to build a timedelta
                return timedelta(
                    hours=t.hour,
                    minutes=t.minute,
                    seconds=t.second,
                    microseconds=t.microsecond,
                )
            except ValueError:
                pass
        raise ValueError("no valid date format found")

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

    def frame_to_secs(self, in_frame):
        comp_frame = in_frame
        if self.override_fps != None and self.mode != "opencv":
            secs = int(in_frame) / self.fps
            override_secs = int(in_frame) / self.override_fps
            if secs != override_secs:
                comp_frame = round(override_secs * self.fps)
                # # DEBUG
                # tqdm.write(
                #     f"Compensating fps-caused frame difference ({self.fps} vs. {self.override_fps}): {override_secs}s -> {secs}s"
                # )
        return int(comp_frame) / self.fps

    def secs_to_frame(self, in_time_in_secs):
        out_frame = int(in_time_in_secs * self.fps)
        if self.override_fps != None and self.mode != "opencv":
            in_frame = out_frame
            out_frame =  int(in_time_in_secs * self.override_fps)
            # # DEBUG
            # tqdm.write(
            #     f"Compensating fps-caused time difference ({self.fps} vs. {self.override_fps}): {in_frame} -> {out_frame}"
            # )
        return out_frame

    def re_encode_with_keyframes(self, kf_list):
        string_list = ",".join(kf_list)
        cmd = f"ffmpeg {self.overwrite_flag} -i {self.video} -force_key_frames {string_list} {self.conversion} {self.tmp_file_path}"
        utils.exec_shell_command(cmd, silent=True)

    def cut(self, **kwargs):
        """Single cut video with parameters

        kwargs:
        from_time (str):    (optional) hh:mm:ss[.micro] (default -> 00:00:00.0)
        from_frame (str):   (optional - alt: from_time)
        to_time (str):      (optional - alt: frames/duration) hh:mm:ss[.micro] (default if no alts given: video length)
        duration (str):     (optional - alt: to_time/frames) hh:mm:ss[.micro] (default if no alts given: video length)
        frames (str):       (optional - alt: to_time/duration) number of frames (default if no alts given: video length)
        out_file (str):     (optional) output file name (default: input file name with timestamps)
        """

        # sanity checks
        if len(kwargs.keys() & {"to_time", "duration", "frames"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "to_time" or "duration" (hh:mm:ss[.micro]) or "frames" (int)'
            )
        if len(kwargs.keys() & {"from_time", "from_frame"}) > 1:
            raise ValueError(
                'ONLY one keyword argument is allowed: "from_time" (hh:mm:ss[.micro]) or "from_frame"'
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
            secs_float = self.frame_to_secs(kwargs["frames"])
            duration_time = self.secs_to_timedelta(secs_float)
            to_time = from_time + duration_time
            # print(f"{utils.get_file_name(self.video)}: f {kwargs["frames"]} fps {self.fps} -> {secs}s")

        # build output file: timestamped of via given parameter
        in_file_name = utils.get_file_name(self.video)
        out_file_ext = utils.get_file_ext(self.video)
        if self.mode == "opencv":
            out_file_ext = f".{self.conversion['container']}"
        out_file = f"{in_file_name}_({str(from_time).replace(':', '-')})-({str(to_time).replace(':', '-')}){out_file_ext}"
        if "out_file" in kwargs.keys():
            kwarg_ext = utils.get_file_ext(kwargs["out_file"])
            if out_file_ext == kwarg_ext:
                out_file = kwargs["out_file"]
            else:
                out_file = utils.replace_right(
                    kwargs["out_file"], kwarg_ext, out_file_ext
                )

        # full out path
        out_path = utils.join_paths(self.out_root, out_file)

        # DEBUG
        # print(f"from\n\t{from_time}")
        # print(f"to\n\t{to_time}")
        # print(f"fps\n\t{self.fps}")
        # print(f"dur\n\t{duration}")
        # print(f"dur\n\t{self.total_duration}")
        # print(f"out {out_file}")

        if self.mode == "ffmpeg":
            self.run_ffmpeg(from_time, to_time, duration_time, out_path)
        elif self.mode == "opencv":
            video_cap = cv2.VideoCapture(self.video)
            # self.fps = video_cap.get(cv2.CAP_PROP_FPS)
            self.run_opencv(video_cap, from_time, to_time, duration_time, out_path)
            video_cap.release()

    def run_ffmpeg(
        self,
        from_time,
        to_time,
        duration_time,
        out_path,
        video_override=None,
        conversion_override=None,
        frame_based=False,
    ):

        conversion = self.conversion
        if conversion_override is not None:
            conversion = conversion_override
        in_video = self.video
        if video_override is not None:
            in_video = video_override

        # 2021: performance improvement -> directly re-encode with cut command
        # re-encode setting from and to as keyframes
        # self.re_encode_with_keyframes([from_time["str"], to_time["str"]])
        # keyframe_list = [from_time["str"], to_time["str"]]
        # keyframe_list_str = ",".join(keyframe_list)

        # TIME-BASED CUT
        # FFMPEG official seeking docs: https://trac.ffmpeg.org/wiki/Seeking
        # some advice on (still accurate?): https://ottverse.com/trim-cut-video-using-start-endtime-reencoding-ffmpeg/

        # faster seeking: -ss before -i, FFMPEG jumps from iframe to iframe, not very accurate  ('to' is the duration_time of the cutout)
        cmd = f'ffmpeg {self.overwrite_flag} -ss {str(from_time)} -i {in_video} -to {str(duration_time)} {conversion} "{out_path}"'

        # slower seeking: supposed to be more accurate (but insiginificant difference) cut using -ss after -i ('to' is the to_time of the cutout)
        # cmd = f'ffmpeg {self.overwrite_flag} -i {in_video} -ss {str(from_time)} -to {str(to_time)} {self.conversion_string} "{out_path}"'

        # FRAME-BASED CUT

        # HACK: frame extraction with duration is not accurate -> use filter
        if frame_based:
            from_time_secs = self.timedelta_to_secs(from_time)
            from_frame = self.secs_to_frame(from_time_secs)
            # from_frame = int(kwargs["from_frame"])
            to_frame = from_frame + (
                # int(kwargs["frames"]) - 1
                self.secs_to_frame(self.timedelta_to_secs(duration_time))
            )
            cmd = f'ffmpeg {self.overwrite_flag} -i {in_video} -an -vf select="between(n\,{from_frame}\,{to_frame}),setpts=PTS-STARTPTS" {conversion} "{out_path}"'

        utils.exec_shell_command(cmd, silent=True)

        # clean up tmp file
        # utils.remove_file(self.tmp_file_path)

    def run_opencv(self, cap, from_time, to_time, duration_time, out_path):
        """[summary]

        Args:
            cap ([type]): [description]
            from_time ([type]): [description]
            to_time ([type]): [description]
            duration_time ([type]): [description]
            out_path ([type]): [description]
            override_fps ([type], optional): Enables setting fixed FPS, pot. circumventing variable FPS. Defaults to None.
        """
        # Get current width of frame
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        # Get current height of frame
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # handle overwrite
        if utils.exists_file(out_path) and self.overwrite_flag == "-n":
            tqdm.write(f"File exists (overwrite = False): {out_path}")
            return

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*self.conversion["code"])
        out = cv2.VideoWriter(out_path, fourcc, self.fps, (int(width), int(height)))

        from_frame = self.secs_to_frame(self.timedelta_to_secs(from_time))
        to_frame = self.secs_to_frame(self.timedelta_to_secs(to_time))
        num_frames = (to_frame - from_frame) + 1  # frames start at 0

        pbar = tqdm(total=num_frames, desc="cut", leave=False)

        current_frame = 0

        # set start frame depending on method
        if self.override_fps is not None:
            real_fps = ecat_tools.get_fps(self.video, True)
            # leave 100 frames margin for error
            from_compensated = round((from_frame / self.fps) * real_fps) - 100
            if from_compensated < 0:
                from_compensated = 0
            current_frame = from_compensated
        else:
            current_frame = from_frame

        # seek to start frame
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # while(True):
        while cap.isOpened():
            # Capture frame-by-frame
            # ret, frame = cap.read()
            ret = cap.grab()  # faster than always decoding all frames
            frame = None

            if ret == True:
                if self.override_fps is not None:
                    # calculate frame using MSEC
                    millis = cap.get(cv2.CAP_PROP_POS_MSEC)
                    secs = millis / 1000.0
                    calc_frame = round(secs * self.fps)
                    if calc_frame >= from_frame and calc_frame <= to_frame:
                        _, frame = cap.retrieve()
                    elif calc_frame < from_frame:
                        pass
                    else:
                        # sequence is extracted, i.e. calc_frame > to_frame
                        break
                else:
                    if current_frame > to_frame:
                        break
                    _, frame = cap.retrieve()

                # frame overlay
                # text = f"FRAME {currentFrame}"
                # frame = opencv_utils.overlay_text(frame, text, x_pos = int(x), y_pos = 20)

                # Saves for video
                if frame is not None:
                    out.write(frame)

                # Display the resulting frame
                # cv2.imshow('frame',frame)
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            current_frame += 1
            if frame is not None:
                pbar.update(1)

        pbar.close()
        out.release()
        cv2.destroyAllWindows()

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
        if self.mode == "opencv":
            in_file_ext = f".{self.conversion['container']}"
        num_seconds = None
        if "seconds" in kwargs.keys():
            num_seconds = float(kwargs["seconds"])
        elif "frames" in kwargs.keys():
            num_seconds = self.frame_to_secs(
                self.get_compensated_frame(kwargs["frames"])
            )

        if self.mode == "ffmpeg":
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
            if self.mode == "ffmpeg":
                # OLD
                # '-to' denotes the duration
                # cmd = f"ffmpeg {self.overwrite_flag} -ss {str(from_time)} -i {self.tmp_file_path} -to {str(duration)} -c copy '{out_path}'"
                # utils.exec_shell_command(cmd, silent=True)
                self.video = self.tmp_file_path  # temp replace video file
                self.run_ffmpeg(
                    from_time,
                    to_time,
                    duration,
                    out_path,
                    video_override=self.tmp_file_path,
                    conversion_override="-c copy",
                )
            elif self.mode == "opencv":
                # TODO: make more efficient instead of opening a capture for every segment
                video_cap = cv2.VideoCapture(self.video)
                # self.fps = video_cap.get(cv2.CAP_PROP_FPS)
                self.run_opencv(video_cap, from_time, to_time, duration)
                video_cap.release()
        # pot clean up tmp file (ffmpeg mode)
        utils.remove_file(self.tmp_file_path)

