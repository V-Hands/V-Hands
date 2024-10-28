import torch
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
import json

totensor = lambda item: torch.tensor(item, dtype=torch.float32)
toname = lambda item: item.split('/')[-1].split('\\')[-1]
toint = lambda item: int(item.replace('clip', ''))
tofloat = lambda item: float(item.cpu())

def loadjs(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def joints2pixel(joints, screen_w, screen_h, H=40, W=70, pad=0):
    joints_pixel = []
    for i in range(21):
        x, y = float(joints[i][0]), float(joints[i][1])
        x = round((x / screen_w) * W - 0.5)
        y = round((y / screen_h) * H - 0.5)
        x, y = x + pad, y + pad
        joints_pixel.append([x, y])
    return joints_pixel

def creat_basemap(H, W, pad):
    padH = H + 2*pad
    padW = W + 2*pad
    basemap = torch.zeros((padH, padW, 2), dtype=torch.float32)
    for i in range(padH):
        basemap[i,:,0] = i
    for i in range(padW):
        basemap[:,i,1] = i
    return basemap.view(1, 1, padH, padW, 2).float()
basemap_global = creat_basemap(40, 72, 28).cuda()

def stdjoints2d2heatmap(stdjoints2d, basemap, sigma, H, W, pad):
    padH = H + 2*pad
    padW = W + 2*pad
    pixeljoints2d = torch.zeros_like(stdjoints2d)
    pixeljoints2d[:,:,0] = padH * stdjoints2d[:,:,0] - 0.5
    pixeljoints2d[:,:,1] = padW * stdjoints2d[:,:,1] - 0.5
    pixeljoints2d = pixeljoints2d.cuda()
    basemap = basemap.clone().detach().repeat(pixeljoints2d.shape[0], 21, 1, 1, 1)
    pixeljoints2d = pixeljoints2d.view(pixeljoints2d.shape[0], 21, 1, 1, 2)
    difmap = basemap - pixeljoints2d
    squaremap = torch.sum(difmap * difmap, dim=4) 
    heatmap = torch.exp(-squaremap / (2 * sigma * sigma))
    return heatmap

def joints2heatmap_(joints_gt, screen_w, screen_h, H, W, pad, sigma):
    rawpixeljoints2d_gt = joints_gt[:,:,:2].flip(2) / torch.tensor([screen_h, screen_w]).view(-1, 1, 2) * torch.tensor([H, W]).view(1, 1, 2)
    rawpixeljoints2d_gt = rawpixeljoints2d_gt + torch.tensor([pad, pad], dtype=torch.float32).view(1, 1, 2)
    stdjoints2d_gt = rawpixeljoints2d_gt / torch.tensor([H + pad*2, W + pad*2], dtype=torch.float32).view(1, 1, 2)
    heatmap_gt = stdjoints2d2heatmap(stdjoints2d_gt, basemap_global, sigma, H, W, pad)
    return heatmap_gt

def joints2heatmap(joints, screen_w, screen_h, pad=30, sigma=3, H=40, W=72):
    heatmaps = joints2heatmap_(joints, screen_w, screen_h, H, W, pad, sigma)
    return heatmaps

def pad_frames(frames, pad, H=40, W=72):
    frames = F.interpolate(frames, (H, W))
    frames[frames < 170./255.] = 0
    frames_pad = torch.zeros((frames.shape[0], 2, frames.shape[2] + 2*pad, frames.shape[3] + 2*pad), dtype=torch.float32)
    frames_pad[:, 0, pad:pad+frames.shape[2], pad:pad+frames.shape[3]] = frames[:,0,:,:]
    frames_pad[:, 1, :, :] = 1.
    frames_pad[:, 1, pad:pad+frames.shape[2], pad:pad+frames.shape[3]] = 0.
    return frames_pad
