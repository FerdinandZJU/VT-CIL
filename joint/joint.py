import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import argparse
from models.resnet import MyResNetsCMC
import torch.backends.cudnn as cudnn
from models.util import adjust_learning_rate, AverageMeter
from torch.nn import functional as F
from dataloader import JointLoader
import torch.optim as optim
import time
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import random


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--model', type=str, default='resnet18t2')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=18)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--dataset', type=str, default='TaG', choices=['TaG', 'OFR'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--fe_learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--model_path', type=str, default=None, help='the model')
    parser.add_argument('--classes_per_step', type=int, default=5)
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()


    args = parser.parse_args()

    args.steps = args.num_classes // args.classes_per_step
    print("total steps: ", args.steps)
    if args.dataset == 'TaG':
        args.data_folder = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/labels/'
        args.dataroot = '/data1/fh/ssd/pythonProject/VT-CIL/data/dataset/'
    elif args.dataset == 'OFR':
        args.data_folder = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/labels'
        args.dataroot = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/dataset'
    else:
        raise ValueError('data_folder is None.')
    
    args.model_path = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/{}/model_5/ckpt_epoch_240.pth'.format(args.dataset)
    if args.data_folder is None:
        raise ValueError('data_folder is None.')


    args.save_path = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/joint/{}'.format(args.dataset)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loader(args, dataset, mode):
    n_data = len(dataset)
    print('number of '+ mode + ': {}'.format(n_data))
    if mode == 'train':
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers!=0 else None, pin_memory=True)
    elif mode == 'test':
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers!=0 else None, pin_memory=True)
    else:
        raise ValueError('Mode is None.')
    return loader

def set_model(args):
    setup_seed(args.seed)
    criterion = nn.CrossEntropyLoss().cuda()
    model = MyResNetsCMC(args=args)
    print("==> loading pre-trained model '{}'".format(args.model_path))
    if not os.path.exists(args.model_path):
        print("model path not existed.")
        sys.exit()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    classifier = nn.Linear(args.feat_dim, args.num_classes)
    print(classifier)
    classifier = classifier.cuda()
    return model, classifier, criterion

def set_optimizer(args, model, mode='classifier'):
    if mode == 'model':
        optimizer = optim.SGD(model.parameters(),
                          lr=args.fe_learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    return optimizer

def model_save(args, name, epoch, model):
    print('Saving {} at Epoch {}'.format(name, epoch), flush=True)
    if name == 'model':
        model_name = '{}_epoch_{}_lr_{}.pth'.format(name, args.epochs, args.fe_learning_rate)
    else:
        model_name = '{}_epoch_{}_lr_{}.pth'.format(name, args.epochs, args.learning_rate)
    path = os.path.join(args.save_path, model_name)
    torch.save(model.state_dict(), path)
    print('{} saved to {}.'.format(name, path))

def top_1_acc_duo(output_touch, output_vision, target):
    norm_touch = F.softmax(output_touch,dim=1)
    norm_vision = F.softmax(output_vision, dim=1)
    norm_mix = norm_touch + norm_vision
    top1_res = norm_mix.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def test_classifier(args):
    current_classifier = nn.Linear(args.feat_dim, args.num_classes)
    classifier_name = 'classifier_epoch_{}_lr_{}.pth'.format(args.epochs, args.learning_rate)
    path = os.path.join(args.save_path, classifier_name)
    print("loading " + path + "...")
    current_classifier.load_state_dict(torch.load(path))
    print("loading successful...")
    return current_classifier
def output_separete(output):
    dim = output.shape[1] // 2
    return output[:, :dim], output[:, dim:]
def train(args, train_dataset):
    print('===================Start training===================')
    print("loading data...")
    train_loader = get_loader(args, train_dataset, mode='train')
    model, classifier, criterion = set_model(args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer_model = set_optimizer(args, model, mode='model')
    optimizer_classifier = set_optimizer(args, classifier)
    print("==> training...")
    for epoch in range(args.epochs):
        model.train()
        classifier.train()
        epoch_avg = 0.0
        adjust_learning_rate(epoch, args, optimizer_model)
        adjust_learning_rate(epoch, args, optimizer_classifier)
        time1 = time.time()
        losses_train = AverageMeter()
        for idx, (feat, target) in enumerate(train_loader):
            optimizer_model.zero_grad()
            optimizer_classifier.zero_grad()
            feat = feat.float()
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if torch.cuda.device_count() > 1:
                feat_vision, feat_touch = output_separete(model.module.forward(feat))
            else:
                feat_vision, feat_touch = output_separete(model(feat))
            output_vision = classifier(feat_vision)
            output_touch = classifier(feat_touch)
            loss_train = criterion(output_vision, target) + criterion(output_touch, target)
            losses_train.update(loss_train.item(), feat.size(0))
            loss_train.backward()
            optimizer_model.step()
            optimizer_classifier.step()
            if idx % args.print_freq == 0:
                print('Epoch [{0}][{1}/{2}]\t'
                    'Loss_sum {loss.val:.4f}\t'
                    'Loss_avg ({loss.avg:.4f})\t'
                    .format(epoch, idx, len(train_loader), loss=losses_train))
                sys.stdout.flush()
            epoch_avg = losses_train.avg
        time2 = time.time()
        print('Epoch:{} train_loss:{:.5f}\t total time {:.2f}'.format(epoch, epoch_avg, time2 - time1), flush=True)
    model_save(args, 'classifier', epoch, classifier)
    print('classifier saved.')
    model_save(args, 'model', epoch, model)
    print('model saved.')


def test(args, dataset):
    print("===================Start testing===================")
    print('==> loading model...')
    model = MyResNetsCMC(args=args)
    current_model_name = 'model_epoch_{}_lr_{}.pth'.format(args.epochs, args.fe_learning_rate)
    path = os.path.join(args.save_path, current_model_name)
    if not os.path.exists(path):
        print("test model path not existed.")
        sys.exit()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    print("loading " + path + "...")

    print('==> loading classifier...')
    classifier = test_classifier(args=args)
    print(classifier)
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    classifier.cuda()
    criterion.cuda()
    model.eval()
    classifier.eval()
    test_loader = get_loader(args, dataset, mode='test')
    all_touch_outputs = torch.Tensor([]).cuda()
    all_vision_outputs = torch.Tensor([]).cuda()
    all_labels = torch.Tensor([]).cuda()
    with torch.no_grad():
        for idx, (feat, target) in enumerate(test_loader):
            feat = feat.float()
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if torch.cuda.device_count() > 1:
                feat_vision, feat_touch = output_separete(model.module.forward(feat))
            else:
                feat_vision, feat_touch = output_separete(model(feat))
            feat_touch = feat_touch.detach()
            feat_vision = feat_vision.detach()
            touch_output = classifier(feat_touch)
            vision_output = classifier(feat_vision)
            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]'.format(idx, len(test_loader)))
                sys.stdout.flush()
            all_touch_outputs = torch.cat((all_touch_outputs, touch_output), dim=0)
            all_vision_outputs = torch.cat((all_vision_outputs, vision_output), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
        duo_top1 = top_1_acc_duo(all_touch_outputs, all_vision_outputs, all_labels)
        print("Testing res: {:.6f}".format(duo_top1))
    return duo_top1
        

def test_each(args, dataset, step):
    print("===================Start testing step {}===================".format(step))
    print('==> loading model...')
    model = MyResNetsCMC(args=args)
    current_model_name = 'model_epoch_{}_lr_{}.pth'.format(args.epochs, args.fe_learning_rate)
    path = os.path.join(args.save_path, current_model_name)
    if not os.path.exists(path):
        print("test model path not existed.")
        sys.exit()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    print("loading " + path + "...")

    print('==> loading classifier...')
    classifier = test_classifier(args=args)
    print(classifier)
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    classifier.cuda()
    criterion.cuda()
    model.eval()
    classifier.eval()
    test_loader = get_loader(args, dataset, mode='test')
    all_touch_outputs = torch.Tensor([]).cuda()
    all_vision_outputs = torch.Tensor([]).cuda()
    all_labels = torch.Tensor([]).cuda()
    with torch.no_grad():
        for idx, (feat, target) in enumerate(test_loader):
            feat = feat.float()
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if torch.cuda.device_count() > 1:
                feat_vision, feat_touch = output_separete(model.module.forward(feat))
            else:
                feat_vision, feat_touch = output_separete(model(feat))
            feat_touch = feat_touch.detach()
            feat_vision = feat_vision.detach()
            touch_output = classifier(feat_touch)
            vision_output = classifier(feat_vision)
            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]'.format(idx, len(test_loader)))
                sys.stdout.flush()
            all_touch_outputs = torch.cat((all_touch_outputs, touch_output), dim=0)
            all_vision_outputs = torch.cat((all_vision_outputs, vision_output), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
        duo_top1 = round(top_1_acc_duo(all_touch_outputs, all_vision_outputs, all_labels), 6)
        print("Testing res for step {}: {}".format(step, duo_top1))
    return duo_top1

def main():
    begin = time.time()
    args = parse_option()
    print(args)
    print('Training start time: {}'.format(datetime.now()))
    setup_seed(args.seed)
    train_dataset = JointLoader(args, mode='train')
    train(args, train_dataset)
    test_dataset = JointLoader(args, mode='test')
    duo_acc_all = test(args, test_dataset)
    duo_acc_each = []
    test_each_dataset = JointLoader(args, mode='test_each')
    for step in range(args.steps):
        test_each_dataset._test_step(step=step)
        duo_acc = test_each(args, test_each_dataset, step)
        duo_acc_each.append(duo_acc)
    print("Testing res for all: {:.6f}".format(duo_acc_all))    
    print("Testing res for each task: {}".format(duo_acc_each))
        
    end = time.time()
    print('Total time used: {}.'.format(end - begin))

if __name__ == '__main__':
    main()