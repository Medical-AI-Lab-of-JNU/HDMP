import random
import argparse
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from initpack.seed_init import place_seed_points
from utils import *


class MetaSliceData_train:
    def __init__(self, datasets, iter_n=100):
        super().__init__()
        self.datasets = datasets
        self.dataset_n = len(datasets)
        self.iter_n = iter_n

    def __len__(self):
        return self.iter_n

    def __getitem__(self, idx):
        dataset = random.sample(self.datasets, 1)[0]
        return dataset.__getitem__(idx)


def meta_data(args):
    tasks = [0, 1, 2, 3]
    target_task = args.target_task
    print('target_task is : {}'.format(target_task))
    train_dataset = []
    tasks.remove(target_task)
    for task in tasks:
        train_dataset.append(TrainLoader(task, args))
    test_dataset = TestLoader(args)
    meta_tr_dataset = MetaSliceData_train(train_dataset)
    return meta_tr_dataset, test_dataset


class TrainLoader(Dataset, ABC):
    def __init__(self, task_idx, args):
        super().__init__()
        self.task_idx = task_idx
        self.target_task = args.target_task
        self.data_dir = args.data_dir
        self.T0_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.task_idx), 'T0_IMG.npy'))
        self.T1_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.task_idx), 'T1_IMG.npy'))
        self.T2_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.task_idx), 'T2_IMG.npy'))
        self.labels = np.load(os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.task_idx), 'MASK.npy'))
        self.slices = args.slices
        self.shot = args.k_shot
        self.imgs_num_3d = self.T0_imgs.shape[0]
        self.img_slices = self.T0_imgs.shape[1]
        self.max_sp = args.max_sp

    def __len__(self):
        return self.imgs_num_3d

    def __getitem__(self, item):
        idx_space = [i for i in range(self.imgs_num_3d)]
        sub_idxs = random.sample(idx_space, self.shot + 1)
        s_sub_idxs = sub_idxs[:self.shot]
        q_sub_idx = sub_idxs[self.shot]

        s_T0_imgs = self.T0_imgs[s_sub_idxs, ...]
        s_T1_imgs = self.T1_imgs[s_sub_idxs, ...]
        s_T2_imgs = self.T2_imgs[s_sub_idxs, ...]
        s_labels = self.labels[s_sub_idxs, ...]

        q_T0_imgs = self.T0_imgs[q_sub_idx, ...]
        q_T1_imgs = self.T1_imgs[q_sub_idx, ...]
        q_T2_imgs = self.T2_imgs[q_sub_idx, ...]
        q_labels = self.labels[q_sub_idx, ...]

        init_seed_list = []
        for i in range(self.shot):
            seed_list = []
            for framid in range(self.slices):
                mask = s_labels[i, framid, :, :, :]  # 1 x H x W
                mask = mask.reshape(-1, mask.shape[-1])     #256x256
                init_seed = place_seed_points(mask, down_stride=8, max_num_sp=self.max_sp, avg_sp_area=4)
                seed_list.append(init_seed.unsqueeze(0))
            s_init_seed1 = torch.cat(seed_list, 0)  # (slice, max_num_sp, 2)
            init_seed_list.append(s_init_seed1.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list, 0)      #(shot, slice, max_num_sp, 2)

        s_T0_imgs, s_T1_imgs, s_T2_imgs, s_labels, q_T0_imgs, q_T1_imgs, q_T2_imgs, q_labels = random_augment(s_T0_imgs,
                                                                                                              s_T1_imgs,
                                                                                                              s_T2_imgs,
                                                                                                              s_labels,
                                                                                                              q_T0_imgs,
                                                                                                              q_T1_imgs,
                                                                                                              q_T2_imgs,
                                                                                                              q_labels)
        sample = {
            's_T0_x': to_tensor(s_T0_imgs),
            's_T1_x': to_tensor(s_T1_imgs),
            's_T2_x': to_tensor(s_T2_imgs),
            's_y': to_tensor(s_labels),

            'q_T0_x': to_tensor(q_T0_imgs),
            'q_T1_x': to_tensor(q_T1_imgs),
            'q_T2_x': to_tensor(q_T2_imgs),
            'q_y': to_tensor(q_labels),
        }
        return sample, s_init_seed


class TestLoader(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.target_task = args.target_task
        self.data_dir = args.data_dir
        self.s_T0_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.target_task), 'T0_IMG.npy'))
        self.s_T1_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.target_task), 'T1_IMG.npy'))
        self.s_T2_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.target_task), 'T2_IMG.npy'))
        self.s_labels = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'training', str(self.target_task), 'MASK.npy'))

        self.q_T0_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'testing', str(self.target_task), 'T0_IMG.npy'))
        self.q_T1_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'testing', str(self.target_task), 'T1_IMG.npy'))
        self.q_T2_imgs = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'testing', str(self.target_task), 'T2_IMG.npy'))
        self.q_labels = np.load(
            os.path.join(self.data_dir, 'Artificial_1', 'testing', str(self.target_task), 'MASK.npy'))
        self.slices = args.slices
        self.shot = args.k_shot
        self.s_imgs_num = self.s_T0_imgs.shape[0]
        self.q_imgs_num = self.q_T0_imgs.shape[0]
        self.img_slices = self.q_T0_imgs.shape[1]
        self.max_sp = args.max_sp

    def __len__(self):
        return self.q_imgs_num

    def __getitem__(self, idx):
        s_idx_space = [i for i in range(self.s_imgs_num)]
        sub_idxs = random.sample(s_idx_space, self.shot)
        s_sub_idxs = sub_idxs
        q_sub_idx = idx
        s_T0_imgs = self.s_T0_imgs[s_sub_idxs, ...]
        s_T1_imgs = self.s_T1_imgs[s_sub_idxs, ...]
        s_T2_imgs = self.s_T2_imgs[s_sub_idxs, ...]
        s_labels = self.s_labels[s_sub_idxs, ...]

        q_T0_imgs = self.q_T0_imgs[q_sub_idx, ...]
        q_T1_imgs = self.q_T1_imgs[q_sub_idx, ...]
        q_T2_imgs = self.q_T2_imgs[q_sub_idx, ...]
        q_labels = self.q_labels[q_sub_idx, ...]
        sample = {
            's_T0_x': to_tensor(s_T0_imgs),
            's_T1_x': to_tensor(s_T1_imgs),
            's_T2_x': to_tensor(s_T2_imgs),
            's_y': to_tensor(s_labels),

            'q_T0_x': to_tensor(q_T0_imgs),
            'q_T1_x': to_tensor(q_T1_imgs),
            'q_T2_x': to_tensor(q_T2_imgs),
            'q_y': to_tensor(q_labels),
        }
        init_seed_list = []
        for i in range(self.shot):
            seed_list = []
            for framid in range(self.slices):
                mask = s_labels[i, framid, :, :, :]  # 1 x H x W
                mask = mask.reshape(-1, mask.shape[-1])  # 256x256
                init_seed = place_seed_points(mask, down_stride=8, max_num_sp=self.max_sp, avg_sp_area=4)
                seed_list.append(init_seed.unsqueeze(0))
            s_init_seed1 = torch.cat(seed_list, 0)  # (slice, max_num_sp, 2)
            init_seed_list.append(s_init_seed1.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list, 0)  # (shot, slice, max_num_sp, 2)
        return sample, s_init_seed

