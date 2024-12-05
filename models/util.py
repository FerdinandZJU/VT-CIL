import numpy as np
def adjust_learning_rate(epoch, args, optimizer, pretrain=False):
    if pretrain:
        steps = np.sum(epoch > np.asarray(args.pretrain_lr_decay_epochs))
        if steps > 0:
            new_lr = args.pretrain_learning_rate * (args.pretrain_lr_decay_rate ** steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            new_lr = args.learning_rate * (args.lr_decay_rate ** steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    meter = AverageMeter()
