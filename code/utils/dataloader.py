import os
import SimpleITK as sitk
from torch.utils import data
import numpy as np
import torch
import random
from monai.transforms import RandGaussianNoise, RandBiasField, RandKSpaceSpikeNoise, RandGaussianSharpen, RandAdjustContrast
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def pre(x, clip_window=None):
    if clip_window is not None:
        x = np.clip(x, clip_window[0], clip_window[1])
        x = (x - np.min(x))/(np.max(x) - np.min(x)) * 2 -1
    else:
        b = np.percentile(x, 99.5) 
        t = np.percentile(x, 00.5) 
        x = np.clip(x, t, b)
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 -1
    return x

class Dataset3D(data.Dataset):
    def __init__(self, dir_):
        super(Dataset3D, self).__init__()
        self.filenames = dir_

    def __getitem__(self, index):
        rootfile = self.filenames[index]

        img_file = os.path.join(rootfile, "image.nii")
        label_file = os.path.join(rootfile, "label.nii")

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

        img = pre(img)
        assert np.min(img)>=-1
        assert np.max(img)<=1

        img = img.astype(np.float32)[np.newaxis, :, :, :]
        label = label.astype(np.uint8)[np.newaxis, :, :, :]

        return img, label

    def __len__(self):
        return len(self.filenames)
    
class Dataset3D_remap(data.Dataset):
    def __init__(self, dir_, rmmax=50, trans_type="Remap"):
        super(Dataset3D_remap, self).__init__()
        self.filenames = dir_
        if trans_type == "Remap":
            self.func = Remap(isremap=True, p=0, rmmax=rmmax)
        else:
            self.func = Bezier_curve(p=1.0)

    def __getitem__(self, index):
        rootfile = self.filenames[index]

        img_file = os.path.join(rootfile, "image.nii")
        label_file = os.path.join(rootfile, "label.nii")

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

        #img = pre(img, [-100, 800])  ##CT
        img = pre(img)
        assert np.min(img) >= -1
        assert np.max(img) <= 1

        img = img.astype(np.float32)[np.newaxis, :, :, :]
        label = label.astype(np.uint8)[np.newaxis, :, :, :]

        img_r = torch.from_numpy(img)
        img_r = self.func(img_r)

        return img, img_r, label

    def __len__(self):
        return len(self.filenames)
    
augmentations = [RandGaussianSharpen(prob=0.3),
                 RandGaussianNoise(prob=0.3),
                 RandBiasField(prob=0.3), RandAdjustContrast(prob=0.3), RandKSpaceSpikeNoise(prob=0.3)
                 ]

def aug_func(data):
    for _ in range(len(augmentations)):
        data = augmentations[_](data)
    return data

def histgram_shift(data):
    num_control_point = random.randint(2,8)
    reference_control_points = torch.linspace(0, 1, num_control_point)
    floating_control_points = reference_control_points.clone()
    for i in range(1, num_control_point - 1):
        floating_control_points[i] = floating_control_points[i - 1] + torch.rand(
            1) * (floating_control_points[i + 1] - floating_control_points[i - 1])
    img_min, img_max = data.min(), data.max()
    reference_control_points_scaled = (reference_control_points *
                                       (img_max - img_min) + img_min).numpy()
    floating_control_points_scaled = (floating_control_points *
                                      (img_max - img_min) + img_min).numpy()
    data_shifted = np.interp(data, reference_control_points_scaled,
                             floating_control_points_scaled)
    return data_shifted

def shuffle_remap(data, ranges = [-1,1], rand_point = [2,50]):
    control_point = random.randint(rand_point[0],rand_point[1])
    distribu = torch.rand(control_point)*(ranges[1]-ranges[0]) + ranges[0]
    distribu, _ = torch.sort(distribu)

    ### --> -1 point1 ... pointN, 1
    distribu = torch.cat([torch.tensor([ranges[0]]),distribu])
    distribu = torch.cat([distribu,torch.tensor([ranges[1]])])
    shuffle_part = torch.randperm(control_point+1)

    new_image = torch.zeros_like(data)
    for i in range(control_point+1):
        target_part = shuffle_part[i]
        min1,max1 = distribu[i],distribu[i+1]
        min2,max2 = distribu[target_part],distribu[target_part+1]
        coord = torch.where((min1 <= data) & (data< max1))
        new_image[coord] = ((data[coord]-min1)/(max1-min1))*(max2-min2)+min2

    if torch.rand(1) < 0.2:
        new_image = -new_image
    if torch.rand(1) < 0.2:
        new_image = torch.from_numpy(histgram_shift(new_image)).to(torch.float32)
    new_image = torch.clamp(new_image,ranges[0],ranges[1])
    if torch.rand(1) < 0.2:
        new_image = torch.clamp(aug_func(new_image),0,1).to(torch.float32)

    return new_image

class Remap(object):
    def __init__(self, isremap=True, p=0.5, rmmax=50):
        self.isremap = isremap
        self.p = p
        self.rmmax = rmmax

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            prob = random.random()
            if prob <= self.p:
                outputs.append(_input)
            elif prob > self.p:
                out = shuffle_remap(_input, ranges=[-1, 1], rand_point=[2, self.rmmax])
                outputs.append(out)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    x_normalized = (x + 1) / 2
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    nonlinear_x_normalized = np.interp(x_normalized, xvals, yvals)
    nonlinear_x = nonlinear_x_normalized * 2 - 1

    return nonlinear_x.astype(float)

class Bezier_curve(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            out = nonlinear_transformation(_input, prob=self.p)
            outputs.append(out)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
