import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import glob
from os.path import join, exists, dirname, basename
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as f
from .extract_data_tools.example_loader_ddd17  import load_files_in_directory, extract_events_from_memmap
from .data_util import  *
import albumentations as A
from PIL import Image


def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
        "test": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))




class DDD17Image(Dataset):
    def __init__(self, root, split='train', augmentation=False, random_crop=False):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "test"]

        self.split = split
        self.augmentation = augmentation
        self.shape = [260, 346]
        self.random_crop = random_crop
        self.shape_crop = [200, 346]
        self.dirs = get_split(data_dirs, split)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))
        print("[DDD17Image]: Found %s segmentation masks for split %s" % (len(self.files), split))

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0.1, 0.8), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Ground truth
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label = np.array(segmentation_mask)
        # Grayscale image
        segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
        segmentation_mask_filename = segmentation_mask_filepath_list[-1]
        filename_id = segmentation_mask_filename.split('_')[-1]
        img_filename = '_'.join(['img', filename_id])
        img_filepath_list = segmentation_mask_filepath_list
        img_filepath_list[-2] = 'imgs'
        img_filepath_list[-1] = img_filename
        img_file = '/'.join(img_filepath_list)
        if not os.path.exists(img_file):
            img_filename = filename_id.zfill(14)
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)[:-60, :] / 255
        # Data augmentation
        if self.augmentation:
            if self.random_crop:
                A_data = self.transform_a_random_crop(image=img, mask=label)
                img = A_data['image']
                label = A_data['mask']
            else:
                A_data = self.transform_a(image=img, mask=label)
                img = A_data['image']
                label = A_data['mask']
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = torch.from_numpy(label).long()
        img_tensor = transforms.Normalize(mean=0.5, std=0.5)(img_tensor)
        return img_tensor, label_tensor

class DDD17Event(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=1, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=True,
                 nr_events_per_data=32000, random_crop=True):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        self.nr_temporal_bins = nr_bins_per_data    # 几个通道
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.random_crop = random_crop
        self.shape_crop = [200, 346]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.dirs = get_split(data_dirs, split)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))
        print("[DDD17Event]: Found %s segmentation masks for split %s" % (len(self.files), split))

        self.img_timestamp_event_idx = {}
        self.event_data = {}

        print("[DDD17Event]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d,self.delta_t_per_data)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0.1, 0.8), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def apply_augmentation(self, transform_a, event_tensor_list, label):
        events = []
        A_data = transform_a(image=event_tensor_list[0][0, :, :].numpy(), mask=label)
        label = A_data['mask']
            # print(label.shape)  # 448,448
        if self.random_crop and self.split == 'train':
            for event_tensor in event_tensor_list:
                    # print(event_tensor.shape)
                mask = torch.zeros((event_tensor.shape[0], self.shape_crop[0], self.shape_crop[1]))
                    # print('mask:',mask.shape)
                events.append(mask)
                # print('events[0].shape1',events[0].shape)
        else:
            events = event_tensor_list
        for k in range(len(event_tensor_list)):
            for j in range(event_tensor_list[k].shape[0]):
                events[k][j,:,:] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=event_tensor_list[k][j, :, :].numpy())['image'])
            # print('events[0].shape2',events[0].shape)
        return events, label        # 返回一个event的list

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label = np.array(segmentation_mask)
        directory = dirname(dirname(segmentation_mask_file))
        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
        t_events, xyp_events = self.event_data[directory]
        # 这里应该是先从index_250.npy里提取出来这段事件。实际上t_events就是读取出来的时间段
        events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration)

        t_ns = events[:, 2] # 24999xxxx，一直是250ms上下。这个t_ns是纳秒为单位的。
        # 这里就是t_events/事件段数，所以改成4帧的话就是第一帧不用，然后用后面4帧，然后最后帧是带label的。
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
        nr_events_loaded = events.shape[0]
        nr_events_temp = nr_events_loaded // self.nr_events_data

        id_end = 0
        event_tensor = None
        event_list = []
        for i in range(self.nr_events_data):
            id_start = id_end

            if self.fixed_duration:
                
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
                # print('id_start',id_start,'t_ns',t_ns,'t_ns[0]',t_ns[0],'delta_t_ns',delta_t_ns)

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded
            event_representation = generate_input_representation(events[id_start:id_end],
                                                                 self.event_representation,
                                                                 self.shape,
                                                                 nr_temporal_bins=self.nr_temporal_bins,
                                                                 separate_pol=self.separate_pol)

            event_representation = torch.from_numpy(event_representation)
            # 第一个event先不用，这个None正好可以规避。
            if event_tensor is None:
                event_tensor = event_representation
            else:
                # event_tensor = torch.cat([event_tensor, event_representation], dim=0)
                event_list.append(event_representation)
        # event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows
        for i in range(len(event_list)):
                event_list[i] = event_list[i][:, :-60, :]
        # img_tensor = None
        if self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            # img = Image.open(img_file)

            img_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            # img_tensor = img_transform(img)
            # img_tensor = img_tensor[:, :-60, :]

        if self.random_crop and self.split == 'train':
            if self.augmentation:
                if self.require_paired_data:
                    event_list,  label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_list,  label)
        else:
            if self.augmentation:
                if self.require_paired_data:
                    event_list,  label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_list,  label)

        label_tensor = torch.from_numpy(label).long()

        if self.require_paired_data:
            # return event_tensor,  label_tensor
            return event_list,  label_tensor

class DDD17Event_with_histogram(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=1, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=True,
                 nr_events_per_data=32000, random_crop=True):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        self.nr_temporal_bins = nr_bins_per_data    # 几个通道
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.random_crop = random_crop
        self.shape_crop = [200, 346]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.dirs = get_split(data_dirs, split)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))
        print("[DDD17Event]: Found %s segmentation masks for split %s" % (len(self.files), split))

        self.img_timestamp_event_idx = {}
        self.event_data = {}

        print("[DDD17Event]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d,self.delta_t_per_data)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.RandomScale(scale_limit=(0.1, 0.8), p=1),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
                A.HorizontalFlip(p=0.5)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def apply_augmentation(self, transform_a, event_tensor_list, label):
        events = []
        A_data = transform_a(image=event_tensor_list[0][0, :, :].numpy(), mask=label)
        label = A_data['mask']
            # print(label.shape)  # 448,448
        if self.random_crop and self.split == 'train':
            for event_tensor in event_tensor_list:
                    # print(event_tensor.shape)
                mask = torch.zeros((event_tensor.shape[0], self.shape_crop[0], self.shape_crop[1]))
                    # print('mask:',mask.shape)
                events.append(mask)
                # print('events[0].shape1',events[0].shape)
        else:
            events = event_tensor_list
        for k in range(len(event_tensor_list)):
            for j in range(event_tensor_list[k].shape[0]):
                events[k][j,:,:] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=event_tensor_list[k][j, :, :].numpy())['image'])
            # print('events[0].shape2',events[0].shape)
        return events, label        # 返回一个event的list

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label = np.array(segmentation_mask)
        directory = dirname(dirname(segmentation_mask_file))
        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
        t_events, xyp_events = self.event_data[directory]
        # 这里应该是先从index_250.npy里提取出来这段事件。实际上t_events就是读取出来的时间段
        events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration)

        t_ns = events[:, 2] # 24999xxxx，一直是250ms上下。这个t_ns是纳秒为单位的。
        # 这里就是t_events/事件段数，所以改成4帧的话就是第一帧不用，然后用后面4帧，然后最后帧是带label的。
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
        nr_events_loaded = events.shape[0]
        nr_events_temp = nr_events_loaded // self.nr_events_data

        id_end = 0
        event_tensor = None
        event_list = []
        for i in range(self.nr_events_data):
            if i > 0:
                previous_id_start = id_start    # 50-100时，计数图就是0-100
            id_start = id_end

            if self.fixed_duration:
                
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
                # print('id_start',id_start,'t_ns',t_ns,'t_ns[0]',t_ns[0],'delta_t_ns',delta_t_ns)

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded
            if i==(self.nr_events_data-1):
                event_representation = generate_input_representation(events[0:id_end],  # 最后一帧用0-250的
                                                                 self.event_representation,
                                                                 self.shape,
                                                                 nr_temporal_bins=self.nr_temporal_bins,
                                                                 separate_pol=self.separate_pol)
            else:
                event_representation = generate_input_representation(events[id_start:id_end],
                                                                 self.event_representation,
                                                                 self.shape,
                                                                 nr_temporal_bins=self.nr_temporal_bins,
                                                                 separate_pol=self.separate_pol)

            event_representation = torch.from_numpy(event_representation)
            # 第一个event先不用，这个None正好可以规避。
            if event_tensor is None:
                event_tensor = event_representation
            else:
                if i==1:
                    event_list.append(event_representation)
                else:
                    event_representation_histogram = generate_input_representation(events=events[id_start:id_end], \
                                                    event_representation='histogram',shape = self.shape)
                    event_representation_histogram = torch.from_numpy(event_representation_histogram).type(torch.FloatTensor).unsqueeze(0)
                    event_representation = torch.cat([event_representation,event_representation_histogram],dim=0)
                # event_tensor = torch.cat([event_tensor, event_representation], dim=0)
                    event_list.append(event_representation)
        # event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows
        for i in range(len(event_list)):
                event_list[i] = event_list[i][:, :-60, :]
        # img_tensor = None
        if self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            # img = Image.open(img_file)

            img_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            # img_tensor = img_transform(img)
            # img_tensor = img_tensor[:, :-60, :]

        if self.random_crop and self.split == 'train':
            if self.augmentation:
                if self.require_paired_data:
                    event_list,  label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_list,  label)
        else:
            if self.augmentation:
                if self.require_paired_data:
                    event_list,  label = self.apply_augmentation(self.transform_a_random_crop,
                                                                              event_list,  label)

        label_tensor = torch.from_numpy(label).long()

        if self.require_paired_data:
            # return event_tensor,  label_tensor
            return event_list,  label_tensor



if __name__ == "__main__":

    ddd17_seg_path = '/DataHDD0/DDD17_Seg/ddd17_seg/data'


    # Event-image pair
    # ddd17_pair = DDD17Event_with_histogram(root=ddd17_seg_path, delta_t_per_data=250,split='test', event_representation='voxel_grid', nr_events_data=5,
    #                         nr_bins_per_data=3,require_paired_data=True, fixed_duration=True, augmentation=False, random_crop=True)
    ddd17_pair = DDD17Events_ori_temporal(root=ddd17_seg_path, delta_t_per_data=250,split='train', event_representation='voxel_grid', nr_events_data=4,
                            nr_bins_per_data=3,require_paired_data=True, fixed_duration=False, augmentation=False,nr_events_per_data=32000, random_crop=True)
    number = ddd17_pair.__len__()
    event_list, label = ddd17_pair[0]

    print(event_list[0].shape, label.shape)
    transform = transforms.ToPILImage()

    idx = 1

    for event in event_list:
        print('event shape:',event.shape)
        # event = transform(event[:3, :, :])  
        event = transform(event[-1, :, :])    
        event.save(f'event_temporal{idx}_histogram.png')
        idx = idx+1

    
    label_img = Image.fromarray(label.numpy().astype(np.uint8))
    label_img.save(f'label.png')

