import torch
import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from zipfile import ZipFile
import random
from utils import *
from torch.utils.data import DataLoader
cfg = load_config()

class VideoDataset(Dataset):
    def __init__(self, data_root, phase, frame_len):
        super(VideoDataset, self).__init__()

        self.frames_list = []
        self.frames_pad_list = []
        self.joints_list = []
        self.heatmaps_list = []
        self.frame_len = frame_len
        self.phase = phase
        self.folder_name_list = []
        self.video_name_list = []
        folders = glob(data_root + '/*')
        folders.sort()
        for folder_id in range(len(folders)):
            src_root = folders[folder_id]
            video_names = [toname(path) for path in glob(src_root + '/*')]
            video_names.sort(key=toint)

            for video_name in video_names:                
                try:
                    frames = []
                    for i in range(len(glob(f'{src_root}/{video_name}/frames/*'))):
                        img = cv2.imread(f'{src_root}/{video_name}/frames/{i}.png')[:,:,:1]
                        frames.append(img.transpose((2, 0, 1)).astype(np.float32) / 255.)
                    frames = totensor(np.array(frames))
                    joints = loadjs(f'{src_root}/{video_name}/joints.json')
                    hand_types = list(joints.keys())
                    for hand_type in hand_types:
                        joints[hand_type] = totensor(joints[hand_type])

                    self.frames_list.append(frames)
                    self.frames_pad_list.append(pad_frames(frames, cfg.pad))
                    self.joints_list.append(joints)
                    self.folder_name_list.append(src_root.split('/')[-1].split('\\')[-1])
                    self.video_name_list.append(video_name)
                except:
                    print('loading failed:', src_root, video_name)

    def __getitem__(self, idx):
        if self.phase == "train":
            frames = self.frames_list[idx].cuda()
            frames_pad = self.frames_pad_list[idx].cuda()
            joints = self.joints_list[idx]
            start_id = random.randint(0, frames.shape[0] - self.frame_len)
            joints_merge = []
            heatmaps_merge = []
            existence_merge = []
            for joint_type in ['Left', 'Right']:
                if joint_type in joints.keys():
                    cur_joints = joints[joint_type][start_id:start_id+self.frame_len]
                    cur_heatmaps = joints2heatmap(cur_joints, cfg.screen_w, cfg.screen_h, pad=cfg.pad)
                    cur_joints = cur_joints.cuda()
                    cur_existence = torch.ones((self.frame_len, 1), dtype=torch.float32, device='cuda')
                else:
                    cur_joints = torch.zeros((self.frame_len, 21, 3), dtype=torch.float32, device='cuda')
                    cur_heatmaps = torch.zeros((self.frame_len, 21, frames_pad.shape[2], frames_pad.shape[3]), dtype=torch.float32, device='cuda')
                    cur_existence = torch.zeros((self.frame_len, 1), dtype=torch.float32, device='cuda')
                joints_merge.append(cur_joints)
                heatmaps_merge.append(cur_heatmaps)
                existence_merge.append(cur_existence)
            joints_merge = torch.cat(joints_merge, dim=1)
            heatmaps_merge = torch.cat(heatmaps_merge, dim=1)
            existence_merge = torch.cat(existence_merge, dim=1)
            return frames[start_id:start_id+self.frame_len], joints_merge, existence_merge,cfg.screen_w, cfg.screen_h, heatmaps_merge, frames_pad[start_id:start_id+self.frame_len]
        else:
            frames = self.frames_list[idx].cuda()
            frames_pad = self.frames_pad_list[idx].cuda()
            joints = self.joints_list[idx]
            folder_name = self.folder_name_list[idx]
            video_name = self.video_name_list[idx]
            joints_merge = []
            existence_merge = []
            for joint_type in ['Left', 'Right']:
                if joint_type in joints.keys():
                    cur_joints = joints[joint_type].cuda()
                    cur_existence = torch.ones((frames.shape[0], 1), dtype=torch.float32, device='cuda')
                else:
                    cur_joints = torch.zeros((frames.shape[0], 21, 3), dtype=torch.float32, device='cuda')
                    cur_existence = torch.zeros((frames.shape[0], 1), dtype=torch.float32, device='cuda')
                joints_merge.append(cur_joints)
                existence_merge.append(cur_existence)
            joints_merge = torch.cat(joints_merge, dim=1)
            existence_merge = torch.cat(existence_merge, dim=1)
            return frames, joints_merge, existence_merge, cfg.screen_w, cfg.screen_h, frames_pad, folder_name, video_name

    def __len__(self):
        return len(self.frames_list)
    
def get_dataloder(data_root, phase, frame_len, batch_size):
    dataset = VideoDataset(data_root, phase, frame_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(True if phase == 'train' else False), num_workers=0, drop_last=(True if phase == 'train' else False))
    return dataloader

