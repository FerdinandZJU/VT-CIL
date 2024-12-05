import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset
class TouchFolderLabel(Dataset):
    def __init__(self, args, transform=None):
        self.dataset = args.dataset
        self.num_class = args.num_class
        self.label_file = args.data_folder
        self.dataroot = args.dataroot
        if self.num_class == 5:
            with open(os.path.join(self.label_file, 'train_5_classes.txt'), 'r') as f:
                data = f.read().splitlines()
        elif self.num_class == 15:
            with open(os.path.join(self.label_file, 'train.txt'), 'r') as f:
                data = f.read().splitlines()
        else:
            print("No txt file to learn.")
            sys.exit()
        self.length = len(data)
        self.target = []
        self.env = data
        self.transform = transform


    def __getitem__(self, index):
        assert index < self.length, 'index_A range error'
        raw, target = self.env[index].strip().split(',')
        target = int(target)
        idx = os.path.basename(raw)
        dir = os.path.join(self.dataroot, raw.split('/')[0])
        if self.dataset == 'TaG':
            A_img_path = os.path.join(dir, 'video_frame', idx)
            A_gelsight_path = os.path.join(dir, 'gelsight_frame', idx)
        elif self.dataset == 'OFR':
            A_img_path = os.path.join(dir, 'vision', idx)
            A_gelsight_path = os.path.join(dir, 'touch', idx)
        elif self.dataset == 'HCT':
            dir = os.path.dirname(raw)
            A_img_path = os.path.join(dir, 'vision', idx)
            A_gelsight_path = os.path.join(dir, 'tactile', idx)
        img_pic = Image.open(A_img_path).convert('RGB')
        gel_pic = Image.open(A_gelsight_path).convert('RGB')
        if self.transform is not None:
            img_pic = self.transform(img_pic)
            gel_pic = self.transform(gel_pic)
        out = torch.cat((img_pic, gel_pic), dim=0)
        return out, target, index

    def __len__(self):
        return self.length