import numpy as np
import torch
from numpy.random import *
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d, BatchNorm3d
from torch.nn import init
import torch.nn as nn

def net_init(net, mode=None):
    if mode == 'kaiming':
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_in')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, 1.0, 0.01)
                init.constant(m.bias, 0)
    elif mode == 'xavier':
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, 1.0, 0.01)
                init.constant(m.bias, 0)
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    init.normal(m.bias, 0.0, 0.01)
            if isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, 1.0, 0.01)
                init.constant(m.bias, 0)

def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines

def adjust_learning_rate(optimizer, epoch,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * 0.99#min(1, 2 - epoch/float(20))#0.95 best
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def to_var(x, requires_grad=False, volatile=False):
    """
        Varialbe type that automatically choose cpu or cuda
        """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def sample_unit_vec(shape, n):
    mean = torch.zeros(shape)
    std = torch.ones(shape)
    dis = torch.distributions.Normal(mean, std)
    samples = dis.sample_n(n)
    samples = samples.view(n, -1)
    samples_norm = torch.norm(samples, 2, 1).view(n, 1)
    samples = samples/samples_norm
    return samples.view(n,*shape)


