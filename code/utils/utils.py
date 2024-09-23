import csv
import math
import numpy as np
import random
import torch
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class LogWriter(object):
    def __init__(self, name, head):
        self.name = name+'.csv'
        with open(self.name, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()

    def writeLog(self, dict):
        with open(self.name, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(dict)
            f.close()

def dice(pre, gt):
    tmp = pre + gt
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pre)
    c = np.sum(gt)
    dice = (2*a)/(b+c+1e-6)
    return dice

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def EMA(model_A, model_B, alpha=0.999):
    for param_B, param_A in zip(model_B.parameters(), model_A.parameters()):
        param_A.data = alpha*param_A.data + (1-alpha)*param_B.data
    return model_A

def adjust_learning_rate(optimizer, epoch, epochs, lr, schedule, is_cos=False):
    if is_cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def K_fold_file_gen(data_list, k, is_shuffle=False):
    assert k > 1
    length = len(data_list)
    fold_size = length // k
    data_numpy = np.array(data_list)
    if is_shuffle:
        index = [i for i in range(0, length)]
        random.shuffle(index)
        data_numpy = data_numpy[index]

    outputlist = list()

    for i in range(0, k):
        idx = slice(i * fold_size, (i + 1) * fold_size)
        if i == k-1:
            idx = slice(i * fold_size, length)
        i_list = data_numpy[idx].tolist()

        outputlist.append(i_list)

    return tuple(outputlist)


def K_fold_data_gen(data_list, i, k):

    valid_list = data_list[i]
    train_list = list()
    for littleseries in range(0, i):
        train_list = train_list + data_list[littleseries]
    for littleseries in range(i + 1, k):
        train_list = train_list + data_list[littleseries]


    return train_list, valid_list

def selfchannel_sim(fe):

    x = fe[0]
    y = fe[0].permute(1, 0)

    x_norm = F.normalize(x, p=2, dim=1)  
    y_norm = F.normalize(y, p=2, dim=0)

    selfdiffusion = torch.matmul(x_norm, y_norm)
    selfdiffusion = selfdiffusion - selfdiffusion.min() + 1e-8
    selfdiffusion = (selfdiffusion + selfdiffusion.permute(1, 0)) / 2.0 
    selfdiffusion /= selfdiffusion.sum(dim=1)

    return selfdiffusion

def selfchannel_loss(srs, tar):

    srs_diffusion = selfchannel_sim(srs)
    tar_diffusion = selfchannel_sim(tar)

    loss = torch.nn.L1Loss(reduction="mean")
    kl_loss = loss(srs_diffusion, tar_diffusion)
    return kl_loss

def crosschannel_sim(srs, tar):

    x = srs[0]
    y = tar[0]
    similarity = F.cosine_similarity(x, y, dim=1)
    return -torch.mean(similarity)

def print_network_para(model):
    print("------------------------------------------")
    print("Network Architecture of Model:")
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print("Number of trainable parameters {0} in Model".format(num_para))
    print("------------------------------------------")

def sigmoid_rampup(current, low_length, max_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if max_length == 0:
        return 1.0
    else:
        current = np.clip(current, low_length, max_length)
        if current == low_length:
            return 0
        else:
            phase = 1.0 - (current - low_length) / (max_length - low_length)
            return float(np.exp(-5.0 * phase * phase))

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 5, 50)
    y = torch.randn(1, 5, 50)
    sim1 = selfchannel_loss(x, y)
