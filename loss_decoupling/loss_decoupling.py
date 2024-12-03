import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import copy
import argparse
from models.resnet import MyResNetsCMC
import torch.backends.cudnn as cudnn
from models.util import adjust_learning_rate, AverageMeter
from torch.nn import functional as F
from dataloader import IncrementalLoader, ExemplarLoader
import torch.optim as optim
import time
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import random
from itertools import cycle

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
    parser.add_argument('--data_folder', type=str, help='path to data')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exem_batch_size', type=int, default=128)
    parser.add_argument('--exem_training_batch_size', type=int, default=8)
    parser.add_argument('--memory_size', type=int, default=150, help='number of examplars')
    
    parser.add_argument('--lambda_1', type=float, default=1.0)
    parser.add_argument('--lambda_2', type=float, default=1.0)
    parser.add_argument('--tips', type=str, default='notips')

    args = parser.parse_args()

    args.steps = args.num_classes // args.classes_per_step
    print("total steps: ", args.steps)
    if args.dataset == 'TaG':
        args.data_folder = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/labels/'
        args.dataroot = '/data1/fh/ssd/pythonProject/VT-CIL/data/dataset/'
    elif args.dataset == 'OFR':
        args.batch_size = 128
        args.data_folder = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/labels'
        args.dataroot = '/data1/fh/ssd/pythonProject/dataset/real_object/mini_objects/dataset' 
    else:
        raise ValueError('data_folder is None.')
    

    args.model_path = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/pretrain/{}/model_5/ckpt_epoch_240.pth'.format(args.dataset)
    if args.data_folder is None:
        raise ValueError('data_folder is None.')
    
    args.save_path = '/data1/fh/ssd/pythonProject/VT-CIL/ArXiv/save/loss_decoupling/{}'.format(args.dataset)
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
    cudnn.benchmark = True

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

def output_separete(output):
    dim = output.shape[1] // 2
    return output[:, :dim], output[:, dim:]


def get_exemplar(args, step):
    dataset = ExemplarLoader(args, step=step)
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.exem_training_batch_size, shuffle=True,
            num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers!=0 else None, drop_last=True, pin_memory=True)
    return loader


def incremental_classifier(args, step):
    previous_classes = step * args.classes_per_step
    previous_classifier = nn.Linear(args.feat_dim, previous_classes)
    previous_classifier_name = 'classifier_epoch_{}_lr_{}_step_{}.pth'.format(args.epochs, args.learning_rate, str(step-1))
    path = os.path.join(args.save_path, previous_classifier_name)
    print("loading " + path + "...")
    previous_classifier.load_state_dict(torch.load(path))
    print("loading successful...")
    weight = previous_classifier.weight.data
    bias = previous_classifier.bias.data
    out_features = previous_classifier.out_features
    current_classes = (step+1) * args.classes_per_step
    current_classifier = nn.Linear(args.feat_dim, current_classes)
    current_classifier.weight.data[:out_features, :] = weight
    current_classifier.bias.data[:out_features] = bias
    return previous_classifier, current_classifier

def model_save(args, name, epoch, step, model):
    print('Saving {} at Epoch {}'.format(name, epoch), flush=True)
    if name == 'model':
        model_name = '{}_epoch_{}_lr_{}_step_{}.pth'.format(name, args.epochs, args.fe_learning_rate, str(step))
    else:
        model_name = '{}_epoch_{}_lr_{}_step_{}.pth'.format(name, args.epochs, args.learning_rate, str(step))
    path = os.path.join(args.save_path, model_name)
    torch.save(model.state_dict(), path)
    print('{} saved to {}.'.format(name, path))

def test_classifier(args,step):
    current_classes = (step + 1) * args.classes_per_step
    current_classifier = nn.Linear(args.feat_dim, current_classes)
    current_classifier_name = 'classifier_epoch_{}_lr_{}_step_{}.pth'.format(args.epochs, args.learning_rate, str(step))
    path = os.path.join(args.save_path, current_classifier_name)
    print("loading " + path + "...")
    current_classifier.load_state_dict(torch.load(path))
    print("loading successful...")
    return current_classifier


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

def set_model(args, step=0):
    criterion = nn.CrossEntropyLoss().cuda()
    model = MyResNetsCMC(args=args)
    if step == 0:
        classifier = nn.Linear(args.feat_dim, args.classes_per_step)
        print(classifier)
        classifier = classifier.cuda()
        print("==> loading pre-trained model '{}'".format(args.model_path))
        if not os.path.exists(args.model_path):
            print("model path not existed.")
            sys.exit()
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt['model'])
        return model, classifier, criterion

    else:
        previous_model_name = 'model_epoch_{}_lr_{}_step_{}.pth'.format(args.epochs, args.fe_learning_rate, str(step-1))
        path = os.path.join(args.save_path, previous_model_name)
        print("==> loading previous model '{}'".format(path))
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
        print('==> loading step_' + str(step-1) + '_classifier...')
        previous_classifier, current_classifier = incremental_classifier(args=args, step=step)
        print("previous_classifier: ",previous_classifier)
        print("current_classifier: ", current_classifier)
        previous_classifier = previous_classifier.cuda()
        previous_classifier.eval()
        current_classifier =  current_classifier.cuda()
        return model, previous_classifier, current_classifier, criterion

def top_1_acc_duo(output_touch, output_vision, target):
    norm_touch = F.softmax(output_touch,dim=1)
    norm_vision = F.softmax(output_vision, dim=1)
    norm_mix = norm_touch + norm_vision
    top1_res = norm_mix.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def gen_exemplar_set(args, model, step, dataset):
    print("==> sampling...")
    current_class_num = args.classes_per_step * (step + 1)
    examplar_per_class = args.memory_size // current_class_num
    
    examplar_per_modality = examplar_per_class // 2
    new_exemplar_touch_path = gen_exemplar_detail(args, model, step, dataset, examplar_per_modality, modality='touch')
    new_exemplar_vision_path = gen_exemplar_detail(args, model, step, dataset, examplar_per_modality, modality='vision')
    new_exemplar_path = np.vstack([np.concatenate([new_exemplar_touch_path[i, :], new_exemplar_vision_path[i, :]]) for i in range(current_class_num)])
    return new_exemplar_path

def gen_exemplar_detail(args, model, step, dataset, examplar_per_class, modality=None):
    model.eval()
    gen_exemplar_loader = torch.utils.data.DataLoader(dataset, batch_size=args.exem_batch_size, num_workers=args.num_workers,
                                     pin_memory=True, drop_last=False, shuffle=False)
    all_features = torch.Tensor([])
    all_labels = torch.Tensor([])
    old_exemplar_class_path = None
    all_ground_path = dataset._dataset()
    if step > 0:
        exemplar_path = 'exemplar.npy'
        path = os.path.join(args.save_path, exemplar_path)
        old_exemplar_class_path = np.load(path, allow_pickle=True)
    all_ground_path_label = np.array([float(sample.split(',')[1]) for sample in all_ground_path])

    with torch.no_grad():
        for idx, (feat, target) in enumerate(gen_exemplar_loader):
            feat = feat.float()
            feat = feat.cuda(non_blocking=True)
            with torch.no_grad():
                if modality == 'vision':
                    if torch.cuda.device_count() > 1:
                        feat_modality, _ = output_separete(model.module.forward(feat))
                    else:
                        feat_modality, _ = output_separete(model(feat))
                elif modality == 'touch':
                    if torch.cuda.device_count() > 1:
                        _, feat_modality = output_separete(model.module.forward(feat))
                    else:
                        _, feat_modality = output_separete(model(feat))
                else:
                    print('modality not exist.')
                    sys.exit()
                feat_modality = feat_modality.detach()
                feat_modality = F.normalize(feat_modality).cpu()
            all_features = torch.cat((all_features, feat_modality), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
    all_features = all_features.numpy()
    all_labels = all_labels.numpy()
    are_elements_equal = np.array_equal(all_labels, all_ground_path_label)
    if not are_elements_equal:
        print('label contents from txt and dataloader are different.')
        sys.exit()
    else:
        print('GoOd..2Yo..')

    current_step_class = np.array(range(args.classes_per_step * step, args.classes_per_step * (step + 1)))
    step_all_ground_path = []
    for class_id in current_step_class:
        class_features = all_features[all_labels == class_id]
        class_labels_path = all_ground_path[all_labels == class_id]
        class_mean_feature = np.mean(class_features, axis=0)
        class_exemplar_labels_path = []
        now_class_mean = np.zeros((1, class_features.shape[-1]))
        for i in range(examplar_per_class):
            if i >= len(class_features):
                class_exemplar_labels_path.append('0,'+str(class_id))
                continue

            x = class_mean_feature - (now_class_mean + class_features) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += class_features[index]
            class_exemplar_labels_path.append(class_labels_path[index])
        step_all_ground_path.append(class_exemplar_labels_path)
    step_all_ground_path = np.array(step_all_ground_path)

    if step == 0:
        new_exemplar_class_path = step_all_ground_path
    else:
        old_exemplar_path_filtered = old_exemplar_class_path[...,:examplar_per_class]
        new_exemplar_class_path = np.concatenate((old_exemplar_path_filtered, step_all_ground_path), axis=0)
    return new_exemplar_class_path


def apply_temperature(outputs, T):
    outputs_T = torch.nn.functional.softmax(outputs / T, dim=1)
    return outputs_T
def new_class_loss(logits, labels, classes):
    mask = torch.zeros_like(logits)
    mask[:, classes] = 1
    logits_masked = logits - (1 - mask) * 1e9
    return F.cross_entropy(logits_masked, labels, reduction='none')

def old_class_loss(logits, labels, old_classes, new_classes):
    old_pre = torch.logsumexp(logits[:, old_classes], dim=1)
    new_pre = torch.logsumexp(logits[:, new_classes], dim=1)
    logits_pre = torch.stack([old_pre, new_pre], dim=-1)
    labels_pre = torch.ones_like(labels)
    return F.cross_entropy(logits_pre, labels_pre, reduction='none')


def train(args, train_dataset, step):
    print('===================Start training===================')
    T = 2
    print("==> "+str(step)+" incremental step...")
    print("loading data...")
    train_loader = get_loader(args, train_dataset, mode='train')
    if step == 0:
        model, classifier, criterion = set_model(args,step=step)
    else:
        model, _, classifier, criterion = set_model(args, step=step)
        print("construct old model...")
        old_model = copy.deepcopy(model)
        old_model.eval()
        exem_loader = get_exemplar(args, step=step)
        exem_loader_iterator = cycle(exem_loader)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step > 0:
            old_model = old_model.cuda()
            old_model = nn.DataParallel(old_model)

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
        C_o = step * args.classes_per_step
        C = 5
        p = 0.1
        for idx, (feat, target) in enumerate(train_loader):
            output_loss = 0.0
            loss_KD_same = 0.0
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
            if step == 0:
                output_loss = criterion(output_vision, target) + criterion(output_touch, target)
            else:
                exem_feat, exem_target = next(exem_loader_iterator)
                exem_feat = exem_feat.cuda(non_blocking=True)
                exem_target = exem_target.cuda(non_blocking=True)
                if torch.cuda.device_count() > 1:
                    exem_feat_vision, exem_feat_touch = output_separete(model.module.forward(exem_feat))
                else:
                    exem_feat_vision, exem_feat_touch = output_separete(model(exem_feat))
                exem_output_vision = classifier(exem_feat_vision)
                exem_output_touch = classifier(exem_feat_touch)
                loss_KD_same = criterion(exem_output_vision, exem_target) + criterion(exem_output_touch, exem_target)

                old_class = torch.arange(step * args.classes_per_step)
                new_class = torch.arange(step * args.classes_per_step, (step+1) * args.classes_per_step)

                vision_old_loss = old_class_loss(output_vision, target, old_class, new_class)
                vision_new_loss = new_class_loss(output_vision, target, new_class)
                vision_loss = p * (args.classes_per_step / C_o) * vision_old_loss.mean() + C * vision_new_loss.mean()

                touch_old_loss = old_class_loss(output_touch, target, old_class, new_class)
                touch_new_loss = new_class_loss(output_touch, target, new_class)
                touch_loss = p * (args.classes_per_step / C_o) * touch_old_loss.mean() + C * touch_new_loss.mean()
                output_loss = vision_loss + touch_loss
            loss_train = args.lambda_1 * output_loss + args.lambda_2 * loss_KD_same
            losses_train.update(loss_train.item(), feat.size(0))
            loss_train.backward()
            optimizer_model.step()
            optimizer_classifier.step()
            if idx % args.print_freq == 0:
                print('Epoch [{0}][{1}/{2}]\t'
                    'Loss_sum {3.val:.4f}\t'
                    'Loss_avg {3.avg:.4f}\t'
                    'output_loss {4:.6f}\t'
                    'loss_KD_same: {5:.6f}\t'
                    .format(epoch, idx, len(train_loader), losses_train, output_loss, loss_KD_same))
                sys.stdout.flush()
            epoch_avg = losses_train.avg
        time2 = time.time()
        print('Epoch:{} train_loss:{:.5f}\t total time {:.2f}'.format(epoch, epoch_avg, time2 - time1), flush=True)

    new_exemplar_class_path = gen_exemplar_set(args, model, step, train_dataset)
    model_save(args, 'classifier', epoch, step, classifier)
    print('classifier saved.')
    model_save(args, 'model', epoch, step, model)
    print('model saved.')
    exemplar_name = 'exemplar.npy'
    path = os.path.join(args.save_path, exemplar_name)
    np.save(path,new_exemplar_class_path)
    print('exemplar saved.')


def forgetting_test(args, dataset, step, duo_best_acc_list):
    print("===================Start testing===================")
    print('==> loading step_' + str(step) + '_model...')
    model = MyResNetsCMC(args=args)
    current_model_name = 'model_epoch_{}_lr_{}_step_{}.pth'.format(args.epochs, args.fe_learning_rate, str(step))
    path = os.path.join(args.save_path, current_model_name)
    if not os.path.exists(path):
        print("test model path not existed.")
        sys.exit()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    print("loading " + path + "...")

    print('==> loading step_' + str(step) + '_classifier...')
    classifier = test_classifier(args=args, step=step)
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

            touch_output = F.softmax(touch_output, dim=-1).detach()
            vision_output = F.softmax(vision_output, dim=-1).detach()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]'.format(idx, len(test_loader)))
                sys.stdout.flush()
            all_touch_outputs = torch.cat((all_touch_outputs, touch_output), dim=0)
            all_vision_outputs = torch.cat((all_vision_outputs, vision_output), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
        duo_top1 = top_1_acc_duo(all_touch_outputs, all_vision_outputs, all_labels)
        print("Incremental step {} Testing res: {:.6f}".format(step, duo_top1))
        old_duo_acc_list = []
        for i in range(step + 1):
            step_class_list = range(i * args.classes_per_step, (i + 1) * args.classes_per_step)
            step_class_idxs = []
            all_labels = all_labels.cpu()
            all_touch_outputs = all_touch_outputs.cpu()
            all_vision_outputs = all_vision_outputs.cpu()
            for c in step_class_list:
                idxs = np.where(all_labels.cpu().numpy() == c)[0].tolist()
                step_class_idxs += idxs
            step_class_idxs = np.array(step_class_idxs)
            i_labels = torch.Tensor(all_labels.numpy()[step_class_idxs])
            i_touch_logits = torch.Tensor(all_touch_outputs.numpy()[step_class_idxs])
            i_vision_logits = torch.Tensor(all_vision_outputs.numpy()[step_class_idxs])
            i_duo_acc = top_1_acc_duo(i_touch_logits, i_vision_logits, i_labels)
            if i == step:
                current_step_duo_acc = i_duo_acc
                print('Current classifier for current accuracy: ' + str(round(current_step_duo_acc, 6)))
            else:
                old_duo_acc_list.append(round(i_duo_acc, 6))
        if step > 0:
            print('===================previous===================')
            print('Current classifier for previous: ' + str(old_duo_acc_list))
            print('Best for previous: ' + str(duo_best_acc_list))
            print('Forgetting for previous: ')
            forgetting_per_step = []
            for i in range(len(duo_best_acc_list)):
                forgetting_per_step.append(round(duo_best_acc_list[i] - old_duo_acc_list[i], 6))
            print(forgetting_per_step)
            duo_forgetting_avg = sum(forgetting_per_step) / len(forgetting_per_step)
            print('Average forgetting: {:.6f}'.format(duo_forgetting_avg))
            for i in range(len(duo_best_acc_list)):
                duo_best_acc_list[i] = max(duo_best_acc_list[i], old_duo_acc_list[i])
        else:
            duo_forgetting_avg = None
        duo_best_acc_list.append(round(current_step_duo_acc, 6))
        return duo_best_acc_list, duo_forgetting_avg

def main():
    begin = time.time()
    args = parse_option()
    print(args)
    setup_seed(args.seed)
    print('Training start time: {}'.format(datetime.now()))
    duo_best_acc_list = []
    step_duo_forgetting_list = []
    train_dataset = IncrementalLoader(args, mode='train')
    test_dataset = IncrementalLoader(args, mode='test')
        
    for step in range(args.steps):
        train_dataset._incremental_step(step=step)
        train(args, train_dataset, step)
        test_dataset._incremental_step(step=step)
        duo_best_acc_list, duo_step_forgetting = forgetting_test(args, test_dataset, step, duo_best_acc_list)
        if duo_step_forgetting is not None:
            step_duo_forgetting_list.append(duo_step_forgetting)
    mean_forgetting = np.mean(step_duo_forgetting_list)
    print('===================overall===================')
    print('Average Forgetting: {:.6f}'.format(mean_forgetting))
    end = time.time()
    print('Total time used: {}.'.format(end - begin))

if __name__ == '__main__':
    main()