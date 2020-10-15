import os
import glob
import numpy as np
import imageio
from torch.utils.data import Dataset

def parse_touch(file):
    with open(file, 'r') as f:
        observation = [float(x) for x in f.readlines()[0].strip().split(' ')]
        return np.array(observation)

def get_image(file):
    """ Load an image from file """
    return imageio.imread(file)

class RobotDataset(Dataset):
    """ My dataset"""

    def __init__(self, dataset_dir='my_dataset/'):

        super(RobotDataset).__init__()
        self.dataset_dir = dataset_dir
        self._load_timestamps()
        self._init_streams()

    def __len__(self):
        """ Return length of dataset """
        return len(self.touch_timestamps)

    def __getitem__(self, idx):

        ts = self.touch_timestamps[idx]
        observation = parse_touch(self.touch_files[idx])
        rgb_idx, depth_idx = self._sync_indices(idx)
        rgb_frame = self.rgb_video.get_data(rgb_idx)
        depth_frame = get_image(self.depth_files[depth_idx])        

        sample = {'ts':ts, 'touch': observation, 'rgb':rgb_frame, 'depth': depth_frame}
        return sample

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
        """ Initialize stream and lists """

        self.depth_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'depth', 'frame-*.png')))
        self.touch_files = sorted(glob.glob(os.path.join(self.dataset_dir, 'touch', 'observation-*.txt')))
        self.rgb_video = imageio.get_reader(os.path.join(self.dataset_dir, 'rgb/video.mp4'), 'ffmpeg')
        self.touch_files_length = len(self.touch_files)
        self.depth_files_length = len(self.depth_files)

        assert self.touch_files_length == self.touch_timestamps.shape[0], "Number of observations != len(timestamps)"
        assert self.depth_files_length == self.depth_timestamps.shape[0], "Number of depth frames != len(timestamps)"

    def _sync_indices(self, idx):
        """ Sync capture indices """

        ts = self.touch_timestamps[idx]
        rgb_idx = (np.abs(ts - self.rgb_timestamps)).argmin()
        depth_idx = (np.abs(ts - self.depth_timestamps)).argmin()
        return rgb_idx, depth_idx