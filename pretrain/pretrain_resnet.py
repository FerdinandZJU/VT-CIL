import os
import faulthandler
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import argparse
from torchvision import transforms
from dataloader import TouchFolderLabel
import torch
import random
import numpy as np
import torch.utils.data
import torch.backends.cudnn as cudnn
from models.util import adjust_learning_rate, AverageMeter
from models.resnet import MyResNetsCMC
from NCE.NCECriterion import NCECriterion
from NCE.NCEAverage import NCEAverage
import time
try:
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for LwF training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=14, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--model', type=str, default='resnet18t2')
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.5)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=512, help='dim of feat for inner product')
    parser.add_argument('--dataset', type=str, default='OFR', choices=['TaG', 'OFR'])
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--classes_per_step', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.dataset == 'TaG':
        args.data_folder = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/labels/'
        args.dataroot = '/data1/fh/ssd/pythonProject/VT-CIL/data/dataset/'
    elif args.dataset == 'OFR':
        args.nce_k = 50
        args.data_folder = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/labels'
        args.dataroot = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/dataset'
    else:
        raise ValueError('data_folder is None.')
    args.model_path = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/{}'.format(args.dataset)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.method = 'softmax' if args.softmax else 'nce'
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.isdir(args.data_folder):
        raise ValueError('data path not exist: {}'.format(args.data_folder))
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

def get_train_loader(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TouchFolderLabel(args, transform=train_transform)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers!=0 else None, pin_memory=True, sampler=train_sampler)
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))
    return train_loader, n_data

def set_model(args, n_data):
    setup_seed(args.seed)
    if args.model.startswith('resnet'):
        model = MyResNetsCMC(args)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))
    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCECriterion(n_data)
    criterion_ab = NCECriterion(n_data)
    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True
    return model, contrast, criterion_ab, criterion_l

def set_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, args):
    setup_seed(args.seed)
    model.train()
    contrast.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()
    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()
        out_feat = model(inputs, mode='pretrain')
        dim = out_feat.shape[1] // 2
        feat_l = out_feat[:,:dim]
        feat_ab = out_feat[:,dim:]
        if torch.isnan(feat_l).any():
            print('nan in feat_l.')
            sys.exit()
        out_l, out_ab = contrast(feat_l, feat_ab, index)
        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        loss = l_loss + ab_loss
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg

def main():
    faulthandler.enable()
    args = parse_option()
    print(args)
    setup_seed(args.seed)
    train_loader, n_data = get_train_loader(args)
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)
    optimizer = set_optimizer(args, model)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            del state

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
