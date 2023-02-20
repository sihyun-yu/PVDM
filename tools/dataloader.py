import os
import os.path as osp
import math
import random
import pickle
import warnings
import glob

import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from tools.video_utils import VideoClips
from torchvision.io import read_video

data_location = '/data'
from utils import set_random_seed
from tools.data_utils import *

import av

class VideoFolderDataset(Dataset):
    def __init__(self,
                 root,
                 train,
                 resolution,
                 path=None,
                 n_frames=16,
                 skip=1,
                 fold=1,
                 max_size=None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,    # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,    # True for evaluating FVD
                 time_saliency=False,
                 sub=False,
                 seed=42,
                 **super_kwargs,         # Additional arguments for the Dataset base class.
                 ):

        video_root = osp.join(os.path.join(root))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = video_root
        name = video_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.annotation_path = os.path.join(video_root, 'ucfTrainTestlist')
        self.classes = list(natsorted(p for p in os.listdir(video_root) if osp.isdir(osp.join(video_root, p))))
        self.classes.remove('ucfTrainTestlist')
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(video_root, class_to_idx, ('avi',), is_valid_file=None)
        video_list = [x[0] for x in self.samples]

        self.video_list = video_list
        """
        if train:
            self.video_list = video_list
        else:
            self.video_list = []
            for p in video_list:
                video = read_video(p)[0]
                if len(video) >= 128:
                    self.video_list.append(p)
        """


        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.video_list)] + [3, resolution, resolution]
        self.num_channels = 3
        self.return_vid = return_vid

        frames_between_clips = skip
        print(root, frames_between_clips, n_frames)
        self.indices = self._select_fold(self.video_list, self.annotation_path,
                                    fold, train)

        self.size = len(self.indices)
        print(self.size)
        random.seed(seed)
        self.shuffle_indices = [i for i in range(self.size)]
        random.shuffle(self.shuffle_indices)

        self._need_init = True

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]

            """
            for p in data:
                if p in video_list == False:
                    data.remove(p)
            """

            selected_files.extend(data)

        """
        name = "train" if not train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)
        """

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution)
        return video

    def __getitem__(self, idx):
        idx = self.shuffle_indices[idx]
        idx = self.indices[idx]
        video = read_video(self.video_list[idx])[0]
        prefix = np.random.randint(len(video)-self.nframes+1)
        video = video[prefix:prefix+self.nframes].float().permute(3,0,1,2)

        return self._preprocess(video), idx

class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,
                 nframes=16,  # number of frames for each video.
                 train=True,
                 interpolate=False,
                 loader=default_loader,  # loader for "sequence" of images
                 return_vid=True,  # True for evaluating FVD
                 cond=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):

        self._path = path
        self._zipfile = None
        self.apply_resize = True

        # classes, class_to_idx = find_classes(path)
        if 'taichi' in path and not interpolate:
            classes, class_to_idx = find_classes(path)
            imgs = make_imagefolder_dataset(path, nframes * 4, class_to_idx, True)
        elif 'kinetics' in path or 'KINETICS' in path:
            if train:
                split = 'train'
            else:
                split = 'val'
            classes, class_to_idx = find_classes(path)
            imgs = make_imagefolder_dataset(path, nframes, class_to_idx, False, split)
        elif 'SKY' in path:
            if train:
                split = 'train'
            else:
                split = 'test'
            path = os.path.join(path, split)
            classes, class_to_idx = find_classes(path)
            if cond:
                imgs = make_imagefolder_dataset(path, nframes // 2, class_to_idx, False, split)
            else:
                imgs = make_imagefolder_dataset(path, nframes, class_to_idx, False, split)
        else:
            classes, class_to_idx = find_classes(path)
            imgs = make_imagefolder_dataset(path, nframes, class_to_idx, False)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + path + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.nframes = nframes
        self.loader = loader
        self.img_resolution = resolution
        self._path = path
        self._total_size = len(self.imgs)
        self._raw_shape = [self._total_size] + [3, resolution, resolution]
        self.xflip = False 
        self.return_vid = return_vid
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.to_tensor = transforms.ToTensor()
        random.shuffle(self.shuffle_indices)
        self._type = "dir"

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
         assert self._type == 'zip'
         if self._zipfile is None:
             self._zipfile = zipfile.ZipFile(self._path)
         return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def _load_img_from_path(self, folder, fname):
        path = os.path.join(folder, fname)
        with self._open_file(path) as f:
            if pyspng is not None and self._file_ext(path) == '.png':
                img = pyspng.load(f.read())
                img = rearrange(img, 'h w c -> c h w')
            else:
                img = self.to_tensor(PIL.Image.open(f)).numpy() * 255 # c h w
        return img

    def __getitem__(self, index):
        index = self.shuffle_indices[index]
        path = self.imgs[index]

        # clip is a list of 32 frames
        video = natsorted(os.listdir(path[0]))

        # zero padding. only unconditional modeling
        if len(video) < self.nframes:
            prefix = np.random.randint(len(video)-self.nframes//2+1)
            clip = video[prefix:prefix+self.nframes//2]
        else:
            prefix = np.random.randint(len(video)-self.nframes+1)
            clip = video[prefix:prefix+self.nframes]
        
        assert (len(clip) == self.nframes or len(clip)*2 == self.nframes)

        vid = np.stack([self._load_img_from_path(folder=path[0], fname=clip[i]) for i in range(len(clip))], axis=0)
        vid = resize_crop(torch.from_numpy(vid).float(), resolution=self.img_resolution) # c t h w 
        if vid.size(1) == self.nframes//2:
            vid = torch.cat([torch.zeros_like(vid).to(vid.device), vid], dim=1)

        return rearrange(vid, 'c t h w -> t c h w'), index


    def __len__(self):
        return self._total_size

def get_loaders(rank, imgstr, resolution, timesteps, skip, batch_size=1, n_gpus=1, seed=42,  cond=False, use_train_set=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    if imgstr == 'UCF101':
        train_dir = os.path.join(data_location, 'UCF-101')
        test_dir = os.path.join(data_location, 'UCF-101') # We use all 
        if cond:
            print("here")
            timesteps *= 2 # for long generation
        trainset = VideoFolderDataset(train_dir, train=True, resolution=resolution, n_frames=timesteps, skip=skip, seed=seed)
        print(len(trainset))
        testset = VideoFolderDataset(train_dir, train=False, resolution=resolution, n_frames=timesteps, skip=skip, seed=seed)
        print(len(testset))
    
    elif imgstr == 'SKY':
        train_dir = os.path.join(data_location, 'SKY')
        test_dir = os.path.join(data_location, 'SKY')
        if cond:
            print("here")
            timesteps *= 2 # for long generation
        trainset = ImageFolderDataset(train_dir, train=True, resolution=resolution, nframes=timesteps, cond=cond)
        print(len(trainset))
        testset = ImageFolderDataset(test_dir, train=False, resolution=resolution, nframes=timesteps, cond=cond)
        print(len(testset))

    else:
        raise NotImplementedError()    

    shuffle = False if use_train_set else True

    kwargs = {'pin_memory': True, 'num_workers': 3}

    trainset_sampler = InfiniteSampler(dataset=trainset, rank=rank, num_replicas=n_gpus, seed=seed)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=4, prefetch_factor=2)
    
    testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus, rank=rank, seed=seed)
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=4, prefetch_factor=2)

    return trainloader, trainloader, testloader 


