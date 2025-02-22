import os
import torch
from torch import nn
from models.network import Seg, UNet_base
from torch.utils.data import DataLoader
from utils.dataloader import Dataset3D as TestDataset
from utils.losses import dice_loss
from utils.utils import AverageMeter, dice, K_fold_data_gen, K_fold_file_gen
import numpy as np

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class DDSPSeg_test(object):
    def __init__(self, args=None):
        super(DDSPSeg_test, self).__init__()

        self.fold_num = args.fold_num
        self.fold = args.fold


        self.direction = args.direction
        self.n_classes = args.num_classes

        A_root = args.A_root
        B_root = args.B_root

        if args.permutationA:
            pA = np.load(args.permutationA)
            pA = pA.tolist()
        else:
            pA = list(range(len(os.listdir(A_root))))
        if args.permutationB:
            pB = np.load(args.permutationB)
            pB = pB.tolist()
        else:
            pB = list(range(len(os.listdir(B_root))))

        self.checkpoint = args.checkpoint
        self.load_epoch = args.load_epoch

        data_listA = np.array(sorted([os.path.join(A_root, x) for x in os.listdir(A_root)]))[pA].tolist()
        data_listA = K_fold_file_gen(data_listA, self.fold_num, is_shuffle=False)
        _, valid_kA = K_fold_data_gen(data_listA, self.fold-1, self.fold_num)

        data_listB = np.array(sorted([os.path.join(B_root, x) for x in os.listdir(B_root)]))[pB].tolist()
        data_listB = K_fold_file_gen(data_listB, self.fold_num, is_shuffle=False)
        _, valid_kB = K_fold_data_gen(data_listB, self.fold-1, self.fold_num)

        # initialize model
        self.enc = UNet_base(input_channels=1).cuda()
        self.seg = Seg(out=self.n_classes).cuda()

        if self.direction == "A2B": 
            test = valid_kB
        elif self.direction == "B2A": 
            test = valid_kA

        # initialize the dataloader
        test_dataset = TestDataset(test)
        self.dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # define loss
        self.L_seg = dice_loss

        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')

    def to_categorical(self, y, num_classes=None):
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

    def load_model(self):
        path = self.checkpoint
        epoch = self.load_epoch
        print("loading model epoch ", str(epoch))
        self.enc.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)),strict=True) #, strict=True
        self.seg.load_state_dict(torch.load('{0}/seg_epoch_{1}.pth'.format(path, epoch)),strict=True)

    def test(self):
        self.enc.eval()
        self.seg.eval()

        loss_list = []
        len_dataloader = len(self.dataloader_test)
        data_test_iter = iter(self.dataloader_test)

        for i in range(len_dataloader):  
            tarimg, tarlabel = data_test_iter.__next__()
            Depth = tarimg.shape[2]

            if torch.cuda.is_available():
                tarimg = tarimg.cuda()
                tarlabel = tarlabel.cuda()

            tarlabel = tarlabel.cpu().numpy()[0][0]
            tarlabel = torch.from_numpy(
                self.to_categorical(tarlabel, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            with torch.no_grad():
                latent_b = self.enc(tarimg)
                pred_mask_b = self.seg(latent_b)
                loss_seg = self.L_seg(pred_mask_b, tarlabel)

            tarlab = tarlabel.cpu().numpy()[0]
            tarseg = self.to_categorical(np.argmax(pred_mask_b[0].cpu().numpy(), axis=0), num_classes=self.n_classes)

            tardice_all = []
            for i in range(self.n_classes - 1):
                tardice_all.append(dice(tarseg[i + 1], tarlab[i + 1]))

            loss_list.append([loss_seg.item(), np.mean(tardice_all)])
        mean_loss, mean_dice = np.mean(loss_list, 0)
        print("Testing loss : ", mean_loss, "Testing dice : ", mean_dice)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Testing Function')

    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--direction', default="A2B")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--checkpoint', default="../checkpoint/mmwhs/xxx/fold_0") 
    parser.add_argument('--load_epoch', type=int, default=500)
    parser.add_argument('--A_root', default="../mmwhs/ct")
    parser.add_argument('--B_root', default="../mmwhs/mr")
    parser.add_argument('--permutationA', default="../mmwhs/ct_list.npy")
    parser.add_argument('--permutationB', default="../mmwhs/mr_list.npy")
    
    args = parser.parse_args()

    testmodel = DDSPSeg_test(args = args)
    testmodel.load_model()
    testmodel.test()


