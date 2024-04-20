import random

import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import re

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, train=True, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.train = train
        self.setup()
        self.samples = self.get_all_samples()
        self.sample = []

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        # if self.train == True:
        #     videos = random.sample(videos, 8)
        #     self.sample = videos
        # a =self.dir[:-6]
        # b = os.path.join(self.dir[:-6], 'flow', '*')
        # flow_videos = glob.glob(os.path.join(self.flow, '*'))
        for idx, video in enumerate(sorted(videos)):
            video_name = video.split('/')[-1].split('\\')[-1]
            dataset_name = video.split('/')[2]  ## debug
            format = '*.tif' if dataset_name == 'ped1' else '*.jpg'
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, format))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['idx'] = idx
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])


    def get_all_samples(self):
        frames = []
        # flow = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        # real_step = self._time_step+3 if self.train else self._time_step
        # if self.train == True:
        data_name = videos[0].split('/')[2]
        #     videos = self.sample
        for video in sorted(videos):
            video_name = video.split('/')[-1].split('\\')[-1]
            # if self.train:
            #     if data_name != 'ped1':
            #         for i in range(1, len(self.videos[video_name]['frame']) - self._time_step//2 - 1):
            #             frames.append(self.videos[video_name]['frame'][i])
            #     else:
            #         for i in range(len(self.videos[video_name]['frame']) - self._time_step // 2 - 3):
            #             frames.append(self.videos[video_name]['frame'][i])
            # else:
            #     if data_name != 'ped1':
            #         for i in range(1, len(self.videos[video_name]['frame']) - self._time_step//2 - 2):
            #             frames.append(self.videos[video_name]['frame'][i])
            #     else:
            #         for i in range(len(self.videos[video_name]['frame']) - self._time_step//2 - 3):
            #             frames.append(self.videos[video_name]['frame'][i])
            if self.train:
                for i in range(1, len(self.videos[video_name]['frame']) - self._time_step//2 - 1):
                    frames.append(self.videos[video_name]['frame'][i])
            else:
                for i in range(1, len(self.videos[video_name]['frame']) - self._time_step//2 - 2):
                    frames.append(self.videos[video_name]['frame'][i])


        return frames

    def __getitem__(self, index):
        # a = self.samples[index].split('/')[-2]
        # dataset_name = self.samples[index].split('/')[2]
        video_name = self.samples[index].split('/')[-1].split('\\')[-2]
        # a = self.samples[index].split('.')[-2].split('\\')[-1]
        frame_name = int(self.samples[index].split('.')[-2].split('\\')[-1])
        # b = self.videos[video_name]['idx']  # int(re.findall(r'\d+', video_name)[0])-1
        # bkg = self.bkg if dataset_name != 'shanghaitech' else self.bkg[b]
        batch_forward = []
        batch_backward = []
        # print('----------------------------------------')
        # print(video_name)
        # print(frame_name)
        #
        # print('----------------------------------------')

        foreground = []
        noisy = []
        if self.train:
            for i in range(0, self._time_step, 2):
                # print(frame_name + i - 1, flush=True)

                image_f = np_load_frame(self.videos[video_name]['frame'][frame_name + i - 1], self._resize_height,
                                          self._resize_width)
                if self.transform is not None:
                    batch_forward.append(self.transform(image_f))
            for j in range(self._time_step-1, -1, -2):
                # print(frame_name + j - 1, flush=True)

                image_b = np_load_frame(self.videos[video_name]['frame'][frame_name + j - 1], self._resize_height,
                                      self._resize_width)

                if self.transform is not None:
                    batch_backward.append(self.transform(image_b))
            return np.concatenate(batch_forward, axis=0), np.concatenate(batch_backward, axis=0)
        else:
            for i in range(self._time_step//2):
                # print(frame_name + i - 1, flush=True)

                image_f = np_load_frame(self.videos[video_name]['frame'][frame_name + i - 1],
                                          self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch_forward.append(self.transform(image_f))
            for j in range(self._time_step-1, 3, -1):
                # print(frame_name + j - 1, flush=True)

                image_b = np_load_frame(self.videos[video_name]['frame'][frame_name + j - 1],
                                          self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch_backward.append(self.transform(image_b))
            gt_pred = np_load_frame(self.videos[video_name]['frame'][frame_name + 3 - 1],
                                          self._resize_height, self._resize_width)
            return np.concatenate(batch_forward, axis=0), np.concatenate(batch_backward, axis=0), self.transform(gt_pred)

    def __len__(self):
        return len(self.samples)
