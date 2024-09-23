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
    '''

    Parameters
    ----------
    data_list : list
        the list of inputdata path
    k : int
        the number of k.
    is_shuffle : bool
        if true the inputdata will be random.

    Returns list
    which covers k lists
    -------

    '''
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
    '''
    只分训练测试
    Parameters
    ----------
    data_path : str
        root path
    data_list : list
        the list of inputdata path
    i : int
        fold number i.
    k : int
        the number of k.

    Returns trainpathlist,validpathlist
    -------

    '''
    trainpathlist = list()
    validpathlist = list()

    valid_list = data_list[i]
    train_list = list()
    for littleseries in range(0, i):
        train_list = train_list + data_list[littleseries]
    for littleseries in range(i + 1, k):
        train_list = train_list + data_list[littleseries]


    return train_list, valid_list

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

def selfchannel_sim(fe):
    '''
        计算特征的维度之间的相似性混淆矩阵
        Params:
    	    fe: 特征（1 * C * len(x))
    	Return:
    		loss: disfussion matrix
    '''
    x = fe[0]
    y = fe[0].permute(1, 0)

    x_norm = F.normalize(x, p=2, dim=1)  # F.normalize只能处理两维的数据，L2归一化
    y_norm = F.normalize(y, p=2, dim=0)
    """
    x_soft = F.softmax(x, 0)   #gsoftmax 已经做过
    y_soft = F.softmax(y, 1)
    """
    selfdiffusion = torch.matmul(x_norm, y_norm)
    selfdiffusion = selfdiffusion - selfdiffusion.min() + 1e-8
    selfdiffusion = (selfdiffusion + selfdiffusion.permute(1, 0)) / 2.0 ## symmetric
    selfdiffusion /= selfdiffusion.sum(dim=1)

    return selfdiffusion

def selfchannel_loss(srs, tar):
    '''
        计算特征的维度之间的相似性混淆矩阵
        Params:
    	    srs: 源域特征（1 * C * len(x))
    	    tar: 目标域特征（1 * C * len(x))
    	Return:
    		loss: selfchannel_loss
    '''
    T = 1

    srs_diffusion = selfchannel_sim(srs)
    tar_diffusion = selfchannel_sim(tar)

    #srs_diag = torch.diag(srs_diffusion)
    #tar_diag = torch.diag(tar_diffusion)

    #srs_diffusion = srs_diffusion - torch.diag_embed(srs_diag)
    #tar_diffusion = tar_diffusion - torch.diag_embed(tar_diag)


    #p_y = F.softmax(tar_diffusion/T, dim=-1)
    #p_x = F.softmax(srs_diffusion/T, dim=-1)
    #kl_loss = (F.kl_div(torch.log(p_x), p_y, reduction='sum') + F.kl_div(torch.log(p_y), p_x, reduction='sum'))/2
    #kl_loss =  (torch.sum(p_x * torch.log(p_x/p_y)) + torch.sum(p_y * torch.log(p_y/p_x))) / 2.0

    loss = torch.nn.L1Loss(reduction="mean")
    kl_loss = loss(srs_diffusion, tar_diffusion)

    #kl_loss = torch.mean(torch.abs((srs_diffusion - tar_diffusion)))
    return kl_loss

def crosschannel_sim(srs, tar):
    '''
        计算源域特征和目标域特征
        Params:
    	    srs: 源域特征（1 * C * len(x))
    	    tar: 目标域特征（1 * C * len(x))
    	Return:
    		loss: crosschannel_sim
    '''
    x = srs[0]
    y = tar[0]
    similarity = F.cosine_similarity(x, y, dim=1)
    return -torch.mean(similarity)
    #return  loss(x_norm, y_norm)

def crosschannel_sim2(srs, tar):
    '''
        计算源域特征和目标域特征
        Params:
    	    srs: 源域特征（1 * C * len(x))
    	    tar: 目标域特征（1 * C * len(x))
    	Return:
    		loss: crosschannel_sim
    '''
    x = srs[0]
    y = tar[0]
    similarity = F.cosine_similarity(x, y, dim=1)
    return -torch.mean(similarity)
    #return  loss(x_norm, y_norm)

def print_network_para(model):
    print("------------------------------------------")
    print("Network Architecture of Model:")
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    #print(model)
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
    print(sim1)
