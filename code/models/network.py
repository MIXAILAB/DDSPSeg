import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers import DoubleConv, Down, Up, Resblock3D, init_weights

class UNet_base(nn.Module):
    def __init__(self, input_channels=1, chs=(8, 16, 32, 64, 32, 16, 8), is_batch=False):
        super(UNet_base, self).__init__()
        if is_batch:
            self.norm = "batch"
        else:
            self.norm = "ins"
        self.inc = DoubleConv(input_channels, chs[0]) 
        self.down1 = Down(chs[0], chs[1], self.norm)
        self.down2 = Down(chs[1], chs[2], self.norm)
        self.down3 = Down(chs[2], chs[3], self.norm)
        self.up1 = Up(chs[3] + chs[2], chs[4], self.norm)
        self.up2 = Up(chs[4] + chs[1], chs[5], self.norm)
        self.up3 = Up(chs[5] + chs[0], chs[6], self.norm)
        self.act = torch.nn.Softmax(1)
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - Z % 16) % 16
        diffY = (16 - Y % 16) % 16
        diffX = (16 - X % 16) % 16
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = x[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2, diffX // 2:X + diffX // 2]
        return out

class Seg(nn.Module):
    def __init__(self, in_channels=8, out=5):
        super(Seg, self).__init__()
        self.in_channels = in_channels
        # downsampling
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=32, norm="ins")
        self.conv2 = Resblock3D(32, norm="ins")
        self.conv3 = Resblock3D(32, norm="ins")
        self.conv4 = nn.Conv3d(32, out, 1, 1, 0)

        self.soft = nn.Softmax(1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        lastfe = self.conv3(x)

        x = self.conv4(lastfe)

        x = self.soft(x)
        
        return x
