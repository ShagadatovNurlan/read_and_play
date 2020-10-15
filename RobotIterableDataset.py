import os
import glob
import cv2
import time
import numpy as np
from torch.utils.data import IterableDataset

def parse_touch(file):
    with open(file, 'r') as f:
        observation = [float(x) for x in f.readlines()[0].strip().split(' ')]
        return np.array(observation)

def get_image(file):
    """ Load an image from file """
    return cv2.imread(file)

class RobotIterableDataset(IterableDataset):
    """ My Dataset """

    def __init__(self, dataset_dir='my_dataset/', simulate_live_mode=False, start=None, end=None):

        super(RobotIterableDataset).__init__()
        self.dataset_dir = dataset_dir
        self.live_mode = simulate_live_mode

        self._load_timestamps()
        self._init_streams()

        self.max_iter = self.touch_timestamps.shape[0]
        if start is not None and start>0:
            self.start = start
        else:
            self.start = 0
        if end is not None and end<self.max_iter:
            self.end = end+1
        else:
            self.end = self.max_iter

        assert self.start < self.end, "Start > End"

    def __iter__(self):
        rgb_idx_pivot, _ = self._sync_indices(0)
        rgb_frame = self._get_rgb_frame()
        for idx in range(self.start, self.end):

            if self.live_mode and idx!=self.start:
                start_time = time.time()
                time_diff_sec = (self.touch_timestamps[idx] - self.touch_timestamps[idx-1])/1000.

            ts = self.touch_timestamps[idx]
            observation = parse_touch(self.touch_files[idx])

            rgb_idx, depth_idx = self._sync_indices(idx)
            depth_frame = get_image(self.depth_files[depth_idx])

            rgb_idx_diff = rgb_idx - rgb_idx_pivot
            if rgb_idx_diff>0:
                for _ in range(rgb_idx_diff):
                    rgb_frame = self._get_rgb_frame()
            rgb_idx_pivot = rgb_idx

            sample = {'ts':ts, 'touch': observation, 'rgb':rgb_frame, 'depth': depth_frame}

            if self.live_mode and idx!=self.start:
                time_passed_sec = (time.time() - start_time)
                if time_diff_sec > time_passed_sec:
                    time.sleep(time_diff_sec - time_passed_sec)
            yield sample

    def _load_timestamps(self):
        """ Load timestamps """

        for capture_type in ['rgb', 'depth', 'touch']:
            if capture_type == 'touch':
                timestamps_file = os.path.join(self.dataset_dir, capture_type, 'per_observation_timestamps.txt')
            else:
                timestamps_file = os.path.join(self.dataset_dir, capture_type, 'per_frame_timestamps.txt')
            timestamps = []
            with open(timestamps_file, 'r') as f:
                if not f:
                    print ("error: open file")
                for line in f.readlines():
                    timestamps.append(float(line.strip()))
            if capture_type == 'rgb':
                self.rgb_timestamps = np.array(timestamps)
            elif capture_type == 'depth':
                self.depth_timestamps = np.array(timestamps)
            elif capture_type == 'touch':
                self.touch_timestamps = np.array(timestamps)

    def _init_streams(self):
        """ Initialize tream and lists """

        self.depth_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'depth', 'frame-*.png')))
        self.touch_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'touch', 'observation-*.txt')))
        self.rgb_capture = cv2.VideoCapture(os.path.join(self.dataset_dir, 'rgb/video.mp4'))
        self.rgb_video_length = int(self.rgb_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.touch_files_length = len(self.touch_files)
        self.depth_files_length = len(self.depth_files)

        assert self.rgb_video_length == self.rgb_timestamps.shape[0], "Number of video frames != len(timestamps)"
        assert self.touch_files_length == self.touch_timestamps.shape[0], "Number of observations != len(timestamps)"
        assert self.depth_files_length == self.depth_timestamps.shape[0], "Number of depth frames != len(timestamps)"

    def _sync_indices(self, idx):
        """ Sync capture indices """

        ts = self.touch_timestamps[idx]
        rgb_idx = min((np.abs(ts - self.rgb_timestamps)).argmin(), self.rgb_timestamps.shape[0])
        depth_idx = min((np.abs(ts - self.depth_timestamps)).argmin(), self.depth_timestamps.shape[0])
        return rgb_idx, depth_idx

    def _get_rgb_frame(self):
        """ Iterate over video """

        ret, rgb_frame = self.rgb_capture.read()
        assert ret, "Frame was not captured!!!"
        return rgb_frame