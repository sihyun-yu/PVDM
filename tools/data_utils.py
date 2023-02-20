import os
import os.path as osp
import math
import random
import pickle
import warnings
import glob

import torch
import torch.nn.functional as F
import zipfile
import PIL.Image
from PIL import Image
from PIL import ImageFile
from einops import rearrange
from torchvision import transforms
import json
import numpy as np
import pyspng

from natsort import natsorted

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')


def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def resize_crop(video, resolution):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop
    Args
        video: a tensor of shape [t, c, h, w] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'
    Returns
        a processed video of shape [c, t, h, w]
    """
    _, _, h, w = video.shape

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    video = video[:, :, cropsize[1]:cropsize[3],  cropsize[0]:cropsize[2]]
    video = F.interpolate(video, size=resolution, mode='bilinear', align_corners=False)

    video = video.permute(1, 0, 2, 3).contiguous()  # [c, t, h, w]
    return video

def make_imageclip_dataset(dir, nframes, class_to_idx, vid_diverse_sampling, split='all'):
    """
    TODO: add xflip
    """
    def _sort(path):
        return natsorted(os.listdir(path))

    images = []
    n_video = 0
    n_clip = 0


    dir_list = natsorted(os.listdir(dir))
    for target in dir_list:
        if split == 'train':
            if 'val' in target: dir_list.remove(target)
        elif split == 'val' or split == 'test':
            if 'train' in target: dir_list.remove(target)

    for target in dir_list:
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            subfolder_path = os.path.join(dir, target)
            for subsubfold in natsorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    i = 1

                    if nframes > 0 and vid_diverse_sampling:
                        n_clip += 1

                        item_frames_0 = []
                        item_frames_1 = []
                        item_frames_2 = []
                        item_frames_3 = []

                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])

                                if i % 4 == 0:
                                    item_frames_0.append(item)
                                elif i % 4 == 1:
                                    item_frames_1.append(item)
                                elif i % 4 == 2:
                                    item_frames_2.append(item)
                                else:
                                    item_frames_3.append(item)

                                if i %nframes == 0 and i > 0:
                                    images.append(item_frames_0) # item_frames is a list containing n frames.
                                    images.append(item_frames_1) # item_frames is a list containing n frames.
                                    images.append(item_frames_2) # item_frames is a list containing n frames.
                                    images.append(item_frames_3) # item_frames is a list containing n frames.
                                    item_frames_0 = []
                                    item_frames_1 = []
                                    item_frames_2 = []
                                    item_frames_3 = []

                                i = i+1
                    else:
                        item_frames = []
                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                # fi is an image in the subsubfolder
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])
                                item_frames.append(item)
                                if i % nframes == 0 and i > 0:
                                    images.append(item_frames)  # item_frames is a list containing 32 frames.
                                    item_frames = []
                                i = i + 1

    return images



def make_imagefolder_dataset(dir, nframes, class_to_idx, vid_diverse_sampling, split='all'):
    """
    TODO: add xflip
    """
    def _sort(path):
        return natsorted(os.listdir(path))

    images = []
    n_video = 0
    n_clip = 0


    dir_list = natsorted(os.listdir(dir))
    for target in dir_list:
        if split == 'train':
            if 'val' in target: dir_list.remove(target)
        elif split == 'val' or split == 'test':
            if 'train' in target: dir_list.remove(target)

    dataset_list = []
    for target in dir_list:
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            subfolder_path = os.path.join(dir, target)
            for subsubfold in natsorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)

                    count = 0
                    valid = False
                    for fi in _sort(subsubfolder_path):
                        if is_image_file(fi):
                            valid = True
                            count += 1
                        else:
                            valid = False
                            break
                        """
                            valid = True
                        """
                    if valid and count >= nframes:
                        valid = True
                    else: 
                        valid = False
                    
                    if valid == True:
                        dataset_list.append((subsubfolder_path, count))

    return dataset_list


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1