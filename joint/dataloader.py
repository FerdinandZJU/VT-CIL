import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class JointLoader(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.save_path = args.save_path
        self.data_folder = args.data_folder
        if self.mode == 'train':
            with open(os.path.join(self.data_folder, 'train.txt'), 'r') as f:
                data = f.read().splitlines()
        elif self.mode == 'test' or mode == 'test_each':
            with open(os.path.join(self.data_folder, 'test.txt'), 'r') as f:
                data = f.read().splitlines()
        self.target = []
        if self.mode == 'test_each':
            self.data = self._test_data(data)
        else:
            self.env = data
            self.length = len(self.env)
        self.transform = self._img_transform()


    def _img_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(self.args.crop_low, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        return transform
    
    
    
    
    def _test_data(self, data):
        _data_splited = []
        for _step in range(self.args.steps):
            _data_sampled = []
            _targets = range(_step * self.args.classes_per_step, (_step + 1) * self.args.classes_per_step)
            for _sample in data:
                _, _target = _sample.strip().split(',')
                if int(_target) in _targets:
                    _data_sampled.append(_sample)
            _data_splited.append(_data_sampled)
        return _data_splited

    def _test_step(self, step):
        self.env = self.data[step]
        self.length = len(self.env)
            

    def __getitem__(self, index):
        assert index < self.length, 'index_A range error'
        raw, target = self.env[index].strip().split(',')
        target = int(target)
        if raw == '0':
            return torch.full((6, 224, 224), 0.01), target
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
        return out, target

    def __len__(self):
        return self.length