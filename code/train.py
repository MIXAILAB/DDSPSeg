import os
from datetime import datetime
import torch
from torch import nn
from models.network import Seg, UNet_base
from utils.STN import SpatialTransformer
from utils.Transform_self import SpatialTransform
from utils.dataloader import Dataset3D_remap as TrainDataset
from torch.utils.data import DataLoader
from utils.losses import dice_loss, prob_entropyloss
from utils.utils import AverageMeter, LogWriter, K_fold_data_gen, K_fold_file_gen, selfchannel_loss, crosschannel_sim
import numpy as np

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class DDSPSeg(object):
    def __init__(self, args=None):
        super(DDSPSeg, self).__init__()

        self.fold_num = args.fold_num
        self.fold = args.fold
        self.start_epoch = args.start_epoch
        self.epoches = args.num_epoch
        self.iters = args.num_iters
        self.save_epoch = args.save_epoch

        self.model_name = args.model_name
        self.direction = args.direction

        self.lr_enc = args.lr_enc
        self.lr_seg = args.lr_seg
        self.bs = args.batch_size
        self.n_classes = args.num_classes
        self.srs_rmmax = args.srs_rmmax
        self.tar_rmmax = args.tar_rmmax
        self.csco = args.csco
        self.scco = args.scco

        A_root = args.A_root
        B_root = args.B_root

        """
        Note that when the dataset is selected as BraTS, 
        the different folds need to be divided based on the patients, and the permutationA and permutationB need to be same.
        """

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

        data_listA = np.array(sorted([os.path.join(A_root, x) for x in os.listdir(A_root)]))[pA].tolist()
        data_listA = K_fold_file_gen(data_listA, self.fold_num, is_shuffle=False)
        train_kA, valid_kA = K_fold_data_gen(data_listA, self.fold-1, self.fold_num)

        data_listB = np.array(sorted([os.path.join(B_root, x) for x in os.listdir(B_root)]))[pB].tolist()
        data_listB = K_fold_file_gen(data_listB, self.fold_num, is_shuffle=False)
        train_kB, valid_kB = K_fold_data_gen(data_listB, self.fold-1, self.fold_num)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  

        self.checkpoint_dir = os.path.join(args.checkpoint_root, self.model_name+ '_' +timestamp + '_' + str(self.direction))
        crt_file(self.checkpoint_dir)
        self.checkpoint_ = self.checkpoint_dir + "/fold_" + str(args.fold)
        crt_file(self.checkpoint_)

        # Data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale_x=(0.75, 1.25),
                                            scale_y=(0.75, 1.25),
                                            scale_z=(0.75, 1.25),
                                            do_translate=True,
                                            trans_x=(-0.1, 0.1),
                                            trans_y=(-0.1, 0.1),
                                            trans_z=(-0.1, 0.1),
                                            do_shear=True,
                                            shear_xy=(-np.pi / 18, np.pi / 18),
                                            shear_xz=(-np.pi / 18, np.pi / 18),
                                            shear_yx=(-np.pi / 18, np.pi / 18),
                                            shear_yz=(-np.pi / 18, np.pi / 18),
                                            shear_zx=(-np.pi / 18, np.pi / 18),
                                            shear_zy=(-np.pi / 18, np.pi / 18),
                                            do_elastic_deform=True,
                                            alpha=(0., 512.),
                                            sigma=(10., 13.))

        # initialize model
        self.enc = UNet_base(input_channels=1).cuda()
        self.seg = Seg(out=self.n_classes).cuda()

        self.opt_e = torch.optim.Adam(self.enc.parameters(), lr=self.lr_enc)
        self.opt_seg = torch.optim.Adam(self.seg.parameters(), lr=self.lr_seg)

        self.stn = SpatialTransformer()

        if self.direction == "A2B": 
            train_srs = train_kA + valid_kA
            train_tar = train_kB
        elif self.direction == "B2A": 
            train_srs = train_kB + valid_kB
            train_tar = train_kA

        # initialize the dataloader
        trainsrs_dataset = TrainDataset(train_srs, rmmax=self.srs_rmmax, trans_type=args.transtype)
        traintar_dataset = TrainDataset(train_tar, rmmax=self.tar_rmmax, trans_type=args.transtype)
        self.dataloader_srstrain = DataLoader(trainsrs_dataset, batch_size=self.bs, shuffle=True)
        self.dataloader_tartrain = DataLoader(traintar_dataset, batch_size=self.bs, shuffle=True)

        # define loss
        self.L_seg = dice_loss

        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')
        self.L_fe_log = AverageMeter(name='L_fe')
        self.L_ent_log = AverageMeter(name='L_ent')
        self.L_consist_log = AverageMeter(name='L_consist')

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

    def forward_enc(self):
        self.latent_a = self.enc(self.srs_img)
        self.latent_a_r = self.enc(self.srs_img_r)
        self.latent_b = self.enc(self.tar_img)
        self.latent_b_r = self.enc(self.tar_img_r)

    def forward_seg(self):
        self.latent_a = self.enc(self.srs_img)
        self.latent_a_r = self.enc(self.srs_img_r)
        self.latent_b = self.enc(self.tar_img)
        self.latent_b_r = self.enc(self.tar_img_r)

        self.pred_mask_real_a = self.seg(self.latent_a)
        self.pred_mask_real_a_r = self.seg(self.latent_a_r)
        self.pred_mask_b = self.seg(self.latent_b)
        self.pred_mask_b_r = self.seg(self.latent_b_r)

    def compute_fea_loss(self):

        ###self-sim loss
        C = self.latent_a.shape[1]

        sc_loss = selfchannel_loss(self.latent_a.detach().reshape(1, C, -1), self.latent_b.reshape(1, C, -1))

        cs_loss = crosschannel_sim(self.latent_a[:, :, self.srs_mask].reshape(1, C, -1), self.latent_a_r[:, :, self.srs_mask].reshape(1, C, -1)) 

        self.fe_loss = self.scco * sc_loss + self.csco * cs_loss

        ### When handling BraTS, remove sc_loss.
        # self.fe_loss = self.csco * cs_loss 

    def compute_seg_loss(self):
        self.loss_ent = prob_entropyloss(self.pred_mask_b)

        ##seg loss
        self.seg_loss = self.L_seg(self.pred_mask_real_a, self.srs_label) + self.L_seg(self.pred_mask_real_a_r, self.srs_label)
        self.target_consist = 0.1 * self.L_seg(self.pred_mask_b, self.pred_mask_b_r)
 
    def train_iterator(self, img1, img1_r, img2, img2_r, img1_label, epoch, iters):
        Depth = img1.shape[2]

        self.srs_img = img1
        self.srs_img_r = img1_r
        self.tar_img = img2
        self.tar_img_r = img2_r

        self.srs_label = img1_label
        ##Encoder forward
        self.forward_enc()
        self.compute_fea_loss()

        self.opt_e.zero_grad()
        self.fe_loss.backward()
        self.opt_e.step()
        
        ###Seg forward
        self.forward_seg()
        self.compute_seg_loss()

        self.opt_e.zero_grad()
        self.opt_seg.zero_grad()
        self.sen_loss = self.seg_loss + self.target_consist + self.loss_ent 
        self.seg_loss.backward()

        self.opt_e.step()
        self.opt_seg.step()

        self.L_seg_log.update(self.seg_loss.data, img1.size(0))
        self.L_ent_log.update(self.loss_ent.data, img1.size(0))
        self.L_consist_log.update(self.target_consist.data, img1.size(0))
        self.L_fe_log.update(self.fe_loss.data, img1.size(0))

    def train_epoch(self, epoch):
        self.enc.train()
        self.seg.train()
        for i in range(self.iters):
            srsimg, srsimg_r, srslabel = next(self.dataloader_srstrain.__iter__())
            tarimg, tarimg_r, _ = next(self.dataloader_tartrain.__iter__())

            if torch.cuda.is_available():
                srsimg = srsimg.cuda()
                srsimg_r = srsimg_r.cuda()
                tarimg = tarimg.cuda()
                tarimg_r = tarimg_r.cuda()
                srslabel = srslabel.cuda()

            # Augment the source image and target image
            mat, code_spa = self.spatial_aug.rand_coords(srsimg.shape[2:])

            srsimg = self.spatial_aug.augment_spatial(srsimg, mat, code_spa)
            srslabel = self.spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()
            srsimg_r = self.spatial_aug.augment_spatial(srsimg_r, mat, code_spa)
            tarimg = self.spatial_aug.augment_spatial(tarimg, mat, code_spa)
            tarimg_r = self.spatial_aug.augment_spatial(tarimg_r, mat, code_spa)

            self.srs_mask = (srslabel[0][0] > 0).detach()

            srslabel = srslabel.cpu().numpy()[0][0]
            srslabel = torch.from_numpy(self.to_categorical(srslabel, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            self.train_iterator(srsimg, srsimg_r, tarimg, tarimg_r, srslabel, epoch, i)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_seg_log.__str__(),
                             self.L_ent_log.__str__(),
                             self.L_consist_log.__str__(),
                             self.L_fe_log.__str__()])
            print(res)

    def checkpoint(self, epoch):
        torch.save(self.enc.state_dict(), '{0}/ec_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))
        torch.save(self.seg.state_dict(), '{0}/seg_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))

    def load_model(self, path, epoch):
        print("loading model epoch ", str(epoch))
        self.enc.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)),strict=True) 
        self.seg.load_state_dict(torch.load('{0}/seg_epoch_{1}.pth'.format(path, epoch)),strict=True)

    def train(self):
        self.trainwriter = LogWriter(name=self.checkpoint_ + "/train_" + self.model_name,
                                     head=["epoch", 'loss_ent', 'loss_consist', 'loss_fe', 'loss_seg', 'loss_all']) 

        for epoch in range(self.epoches - self.start_epoch):
            self.L_seg_log.reset()
            self.L_ent_log.reset()
            self.L_consist_log.reset()
            self.L_fe_log.reset()

            self.epoch = epoch
            self.train_epoch(epoch + self.start_epoch)
            loss_all = self.L_ent_log.avg.item() + self.L_consist_log.avg.item() + self.L_fe_log.avg.item() + self.L_seg_log.avg.item()
            self.trainwriter.writeLog([epoch + self.start_epoch, self.L_ent_log.avg.item(), self.L_consist_log.avg.item(), 
                                       self.L_fe_log.avg.item(), self.L_seg_log.avg.item(), loss_all])

            if epoch % self.save_epoch == 0:
                self.checkpoint(epoch)

        self.checkpoint(self.epoches - self.start_epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Training Function')

    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--direction', default="A2B")
    parser.add_argument('--srs_rmmax', type=int, default=30)
    parser.add_argument('--tar_rmmax', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--num_iters', type=int, default=20)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--model_name', default="DDSPSeg")

    parser.add_argument('--lr_enc', type=int, default=1e-4)
    parser.add_argument('--lr_seg', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--A_root', default="../mmwhs/ct")
    parser.add_argument('--B_root', default="../mmwhs/mr")
    parser.add_argument('--permutationA', default="../mmwhs/ct_list.npy")
    parser.add_argument('--permutationB', default="../mmwhs/mr_list.npy")

    parser.add_argument('--transtype', type=str, choices = ['Remap', 'BC'], default='Remap')
    parser.add_argument('--csco', type=float, default=0.1)
    parser.add_argument('--scco', type=float, default=0.1)
    parser.add_argument('--checkpoint_root', default="./checkpoint/mmwhs") 
    
    args = parser.parse_args()

    trainer = DDSPSeg(args = args)
    trainer.train()

