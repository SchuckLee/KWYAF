"""
Adapted from https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/sequence.py
"""

from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from joblib import Parallel, delayed
from .eventslicer import EventSlicer
import albumentations as A
from ...data_util import generate_input_representation
# from .data_util_dsec import generate_input_representation

import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)


class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   └──data
    # │   │   │       ├── 000000.png
    # │   │   │       └── ...
    # │   │   └── 19classes
    # │   │       └──data
    # │   │           ├── 000000.png
    # │   │           └── ...
    # │   └── timestamps.txt
    # ├── events
    # │   └── left
    # │       ├── events.h5
    # │       └── rectify_map.h5
    # └── images
    #     └── left
    #         ├── rectified
    #         │    ├── 000000.png
    #         │    └── ...
    #         ├── ev_inf
    #         │    ├── 000000.png
    #         │    └── ...
    #         └── timestamps.txt

    def __init__(self, seq_path: Path, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=False, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=False, fixed_duration=False, remove_time_window: int = 250,
                 random_crop=False):
        assert nr_bins_per_data >= 1
        assert seq_path.is_dir()
        self.sequence_name = seq_path.name
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.random_crop = random_crop
        # self.shape_crop = [480,480]
        # self.shape_crop = [352, 512]
        self.shape_crop = [440,640]
        # self.shape_crop = [448,448]

        # Set event representation
        self.nr_events_data = nr_events_data        # 1 每个事件窗口中包含的事件数量
        self.num_bins = nr_bins_per_data            # 5 时间轴被分成的时间间隔数量
        self.nr_events_per_data = nr_events_per_data    # 3  事件数据块（或者称为数据样本）中包含的事件窗口数量
        self.event_representation = event_representation    # voxel_grid
        self.separate_pol = separate_pol    # 不把极性分为两个通道
        self.normalize_event = normalize_event
        self.locations = ['left']
        self.semseg_num_classes = semseg_num_classes
        self.augmentation = augmentation
        # fixed_duration = True   # 写错了，这个应该是在main函数里定义的。
        # print('fixed_duration',fixed_duration)
        # Save delta timestamp
        self.fixed_duration = fixed_duration # 是否采用固定时长的事件窗口（True）
        if self.fixed_duration:
            # delta_t_ms = nr_events_data * delta_t_per_data  # delta_t_ms代表了固定时长的事件窗口的时长，以毫秒为单位 在这里，200ms变成了800ms
            delta_t_ms =  delta_t_per_data
            self.delta_t_us = delta_t_ms * 1000 # 微秒为单位
            # print('delta_t_ms',delta_t_ms) # 50
            # print('delta_t_us',self.delta_t_us) # 50000
        self.remove_time_window = remove_time_window

        self.require_paired_data = require_paired_data

        # load timestamps
        self.timestamps = np.loadtxt(str(seq_path / 'semantic' /'left'/ 'timestamps.txt'), dtype='int64')

        # load label paths
        if self.semseg_num_classes == 11:
            label_dir = seq_path / 'semantic'  /'left'/ '11classes' 
        elif self.semseg_num_classes == 19:
            label_dir = seq_path / 'semantic' /'left' / '19classes' 
        else:
            raise ValueError
        assert label_dir.is_dir()
        label_pathstrings = list()
        for entry in label_dir.iterdir():
            assert str(entry.name).endswith('.png')
            label_pathstrings.append(str(entry))
        label_pathstrings.sort()
        self.label_pathstrings = label_pathstrings
        assert len(self.label_pathstrings) == self.timestamps.size  # 时间戳的个数和label的个数相等

        # load images paths
        if self.require_paired_data:
            img_dir = seq_path / 'images'
            img_left_dir = img_dir / 'left' / 'ev_inf'
            assert img_left_dir.is_dir()
            img_left_pathstrings = list()
            for entry in img_left_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_left_pathstrings.append(str(entry))
            img_left_pathstrings.sort()
            self.img_left_pathstrings = img_left_pathstrings
            assert len(self.img_left_pathstrings) == self.timestamps.size

        # Remove several label paths and corresponding timestamps in the remove_time_window (i.e., the first six samples).
        # This is necessary because we do not have enough events before the first label.数据集的起始部分，没有足够的事件数据来生成第一个标签，所以需要移除这些数据以确保数据的一致性。
        self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
        del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        assert len(self.label_pathstrings) == self.timestamps.size
        if self.require_paired_data:
            del self.img_left_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
            assert len(self.img_left_pathstrings) == self.timestamps.size

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        # Data Augmentation Configuration
        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0.1, 0.5), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256

    @staticmethod
    def get_img(filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = img_transform(img)
        return img_tensor

    @staticmethod
    def get_label(filepath: Path):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return (self.timestamps.size + 1) // 2

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert np.max(x) < self.width
        assert np.max(y) < self.height
        return rectify_map[y, x]

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = generate_input_representation(events_temp, self.event_representation,
                                                                       (self.height, self.width),
                                                                        nr_temporal_bins=self.num_bins,
                                                                        separate_pol=self.separate_pol)
        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)
        num_bins_temp = event_representation.shape[0]
        # print('num_bins_temp',num_bins_temp)
        event_tensor[(job_id * num_bins_temp):((job_id+1) * num_bins_temp), :, :] = event_representation
    def generate_event_histogram(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = generate_input_representation(events_temp, 'histogram',
                                                                       (self.height, self.width))
        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)
        num_bins_temp = 1
        event_tensor[(job_id * num_bins_temp):((job_id+1) * num_bins_temp), :, :] = event_representation


    def apply_augmentation(self, transform_a, event_tensor_list, images, label):
        events = []
        if self.require_paired_data:
            A_data = transform_a(image=images.permute(1, 2, 0).numpy(), mask=label)
            img_tensor = torch.from_numpy(A_data['image']).permute(2, 0, 1)
            label = A_data['mask']
            if self.random_crop and self.mode == 'train':
                events = [torch.zeros((event_tensor.shape[1], self.shape_crop[0], self.shape_crop[1])) for event_tensor in event_tensor_list]
            else:
                events = event_tensor_list
            for k in range(len(event_tensor_list)):
                for j in range(event_tensor_list[k].shape[0]):
                    
                    events[k][j,:,:] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=event_tensor_list[k][j, :, :].numpy())['image'])
                # events[k][:, :, :] = torch.from_numpy(
                # A.ReplayCompose.replay(A_data['replay'], image=event_tensor_list[k][0, :, :].numpy())['image'])
            return events, img_tensor, label
        else:

            A_data = transform_a(image=event_tensor_list[0][0, :, :].numpy(), mask=label)
            
            label = A_data['mask']
            # print(label.shape)  # 448,448
            if self.random_crop and self.mode == 'train':
                for event_tensor in event_tensor_list:
                    mask = torch.zeros((event_tensor.shape[0], self.shape_crop[0], self.shape_crop[1]))
                    events.append(mask)

            else:
                events = event_tensor_list
            for k in range(len(event_tensor_list)):
                for j in range(event_tensor_list[k].shape[0]):
                    # if j==3:
                    #     events[k][j, :, :] = event_tensor_list[k][j, :, :]
                    events[k][j,:,:] = torch.from_numpy(
                    A.ReplayCompose.replay(A_data['replay'], image=event_tensor_list[k][j, :, :].numpy())['image'])
        return events, label        # 返回一个event的list
    # """

    def __getitem__(self, index):
        output = {}
        # Load the ground truth
        label_path = Path(self.label_pathstrings[index * 2])   
        
        label = self.get_label(label_path)
        # Load the events and generate a frame-based representation
        ts_end = self.timestamps[index * 2]     # timestamps是加载的txt文件
        # event_tensor = None
        event_tensor = []
        # u = 0     
        for location in self.locations:
            # u = u + 1
            # print('location:',location) # location: left； 这个for循环只会遍历一次。
            if self.fixed_duration: # 是否使用固定时长
                ts_start = ts_end - self.delta_t_us         # ts_start是end减去了间隔，但是label是从end的时候取的，所以target还真是150-200的帧。
                self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data # 这些微秒内一个事件的时间（nr=1）

                for i in range(self.nr_events_data):


                    t_s = ts_start + i * self.delta_t_per_data_us       # 每隔50ms堆叠一次
                    t_end = ts_start + (i+1) * self.delta_t_per_data_us


                    event_data = self.event_slicers[location].get_events(t_s, t_end)
                    p = event_data['p']
                    t = event_data['t']
                    x = event_data['x']
                    y = event_data['y']
                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    if self.event_representation == 'voxel_grid':
                        events = np.stack([x_rect, y_rect, t, p], axis=1)
                        
                        event_representation = generate_input_representation(events, self.event_representation,
                                                                                       (self.height, self.width),
                                                                                        nr_temporal_bins=self.num_bins,
                                                                                        separate_pol=self.separate_pol ) 
                        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)       
                        if i!=0:
                            event_data = self.event_slicers[location].get_events(t_s-self.delta_t_per_data_us, t_end)
               
                            p = event_data['p']
                            t = event_data['t']
                            x = event_data['x']
                            y = event_data['y']

                            xy_rect = self.rectify_events(x, y, location)
                            x_rect = xy_rect[:, 0]
                            y_rect = xy_rect[:, 1]
                            events = np.stack([x_rect, y_rect, t, p], axis=1)
                            event_representation_histogram = generate_input_representation(events=events, event_representation='histogram',
                                                                  shape = (self.height, self.width))
                            event_representation_histogram = torch.from_numpy(event_representation_histogram).type(torch.FloatTensor).unsqueeze(0)
                            event_representation = torch.cat([event_representation,event_representation_histogram],dim=0)
                        
                    else:
                        events = np.stack([x_rect, y_rect, t, p], axis=1)
                        event_representation = generate_input_representation(events, self.event_representation,
                                                                  (self.height, self.width))
                        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)

                    if event_tensor is None:
                        event_tensor.append(event_representation)  
                    else:
                        event_tensor.append(event_representation)
                        


            else:   # 不使用固定时长
                event_tensor = []   # event tensor的list
                for i in range(self.nr_events_data):
                    event_tensor_temp = torch.zeros((self.num_bins, self.height, self.width))
                    event_histogram = torch.zeros((1, self.height, self.width))
                    # self.nr_events = self.nr_events_data * self.nr_events_per_data
                    self.nr_events = self.nr_events_per_data
                    
                    event_data = self.event_slicers[location].get_events_fixed_num(ts_end-((3-i)*self.nr_events), self.nr_events)
                    
                    if self.nr_events >= event_data['t'].size:
                        start_index = 0
                    elif (i==3):
                        start_index = start_index-4*self.nr_events    
                    else:
                        start_index = -self.nr_events

                    p = event_data['p'][start_index:]
                    t = event_data['t'][start_index:]
                    x = event_data['x'][start_index:]
                    y = event_data['y'][start_index:]
                    nr_events_loaded = t.size

                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]
                    events = np.stack([x_rect, y_rect, t, p], axis=-1)

                    nr_events_temp = nr_events_loaded 
                    Parallel(n_jobs=8, backend="threading")(
                        delayed(self.generate_event_tensor)(i, events, event_tensor_temp, nr_events_temp)
                        for i in range(1))
                    if(i==0):
                        event_tensor.append(event_tensor_temp)
                    else:
                        Parallel(n_jobs=8, backend="threading")(
                            delayed(self.generate_event_histogram)(i, events, event_histogram, nr_events_temp)
                            for i in range(1))
                        event_tensor_temp = torch.cat([event_tensor_temp,event_histogram],dim=0)
                        event_tensor.append(event_tensor_temp)

            # Remove 40 bottom rows
            # event_tensor = event_tensor[:, :-40, :]
            for i in range(len(event_tensor)):
                event_tensor[i] = event_tensor[i][:, :-40, :]

        # Generate the event-image pair
        # print('u=',u)
        img_tensor = None
        if self.require_paired_data:
            img_left_path = Path(self.img_left_pathstrings[index * 2])
            img_tensor = self.get_img(img_left_path)[:, :-40, :]

        # Data augmentation
        if self.random_crop and self.mode == 'train':
            if self.augmentation:
                if self.require_paired_data:
                    event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_tensor, img_tensor, label)
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor,
                                                                  img_tensor, label)
        else:
            if self.augmentation:
                if self.require_paired_data:
                    event_tensor, img_tensor, label = self.apply_augmentation(self.transform_a, event_tensor,
                                                                              img_tensor, label)
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, img_tensor, label)

        label_tensor = torch.from_numpy(label).long()
        if 'representation' not in output:
            output['representation'] = dict()
        output['representation']['left'] = event_tensor
        output['img_left'] = img_tensor

        if self.require_paired_data:
            return output['representation']['left'], output['img_left'], label_tensor   # False
        else:
            return output['representation']['left'], label_tensor
