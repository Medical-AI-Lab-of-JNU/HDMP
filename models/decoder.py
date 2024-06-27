import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch, momentum=1, affine=True),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(2 * in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch, momentum=1, affine=True),
            nn.ReLU()
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(192, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2, momentum=1, affine=True),
            nn.ReLU()
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(96, in_ch // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 4, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_ch // 4, in_ch // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 4, momentum=1, affine=True),
            nn.ReLU()
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(48, in_ch // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 8, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_ch // 8, 1, kernel_size=1, padding=0),  # 1 for bce and 2 for cross entropy loss
            nn.Sigmoid()
        )  # 112 x 112
        # self.double_conv5 = nn.Sequential(
        #     nn.Conv2d(in_ch // 4, in_ch // 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(in_ch // 8, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.Conv2d(in_ch // 8, 1, kernel_size=1, padding=0),  # 1 for bce and 2 for cross entropy loss
        #     nn.Sigmoid()
        # )  # 256 x 256
        self._init_weights()

    # [-1,64,256,256]
    # [-1,128,128,128]
    # [-1,256,64,64]
    # [-1,512,32,32]
    # [-1,512,32,32]   扩张卷积
    def forward(self, hidden, ft_list):
        """
        hidden.shape = (B,c,h,w)  c=512
        ft_list   每层卷积的特征
        """
        out = self.layer1(hidden)  #128*32*32
        out = self.layer2(out)  #128*32*32
        # out = self.upsample(out)  # block 1
        out = torch.cat((out, ft_list[-1]), dim=1)  #256*32*32
        out = self.double_conv1(out)  #128*32*32
        out = self.upsample(out)  #128*64*64
        out = torch.cat((out, ft_list[-2]), dim=1)  #192*32*32
        out = self.double_conv2(out)  #64*64*64
        out = self.upsample(out)  # block 3   64*128*128
        out = torch.cat((out, ft_list[-3]), dim=1)  #96*128*128
        out = self.double_conv3(out)  #32*128*128
        out = self.upsample(out)  # block 4    32*256*256
        out = torch.cat((out, ft_list[-4]), dim=1)  #48*256*256
        out = self.double_conv4(out)  #1*256*256
        # out = self.upsample(out)  # block 5    16*256*256
        # out = torch.cat((out, ft_list[-5]), dim=1)  #32*256*256
        # out = self.double_conv5(out)  #1*256*256
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def forward(self, x):
        x = self.conv(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
