from torchvision.io import read_video
import numpy as np


def sample_video(filename: str, num_sample: int = 5):
    video_frames, _, info = read_video(filename, pts_unit='sec')
    sampled_frame_indexes = np.linspace(0, video_frames.size(0) - 1, num_sample).round().astype(int)
    sampled_images = video_frames[sampled_frame_indexes]
    return sampled_images