from torch import nn
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length), 32)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(128, 256)

        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.csa1 = CSA(256)
        self.moduleConv4 = Basic(256, 256)
        self.moduleUpsample4 = Upsample(256, 128)

        self.csa2 = CSA(128)
        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128, 64)

        self.csa3 = CSA(64)
        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64, 32)

        self.csa4 = CSA(32)
        self.moduleDeconv1 = Gen(64, n_channel, 32)

    def forward(self, x, skip1, skip2, skip3):
        x = self.csa1(x)
        tensorConv = self.moduleConv4(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        
        tensorUpsample4 = self.csa2(tensorUpsample4)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        
        tensorUpsample3 = self.csa3(tensorUpsample3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        
        tensorUpsample2 = self.csa4(tensorUpsample2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)
        output = self.moduleDeconv1(cat2)

        return output


class CA(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        x_sig = self.sigmoid(avg_out)
        x_mul = x * x_sig
        return x_mul


class Res(nn.Module):
    def __init__(self, in_planes, ks1=3, ks2=5):
        super(Res, self).__init__()
        assert ks1 in (1, 3, 5, 7), 'kernel size must be 1, 3, 5 or 7'
        assert ks1 in (1, 3, 5, 7), 'kernel size must be 1, 3, 5 or 7'

        pd1 = (ks1 - 1) // 2
        pd2 = (ks2 - 1) // 2
        self.c1_1 = nn.Conv2d(in_planes, in_planes, ks1, padding=pd1, bias=False)
        self.relu1 = nn.ReLU()
        self.c1_2 = nn.Conv2d(in_planes, in_planes, ks2, padding=pd2, bias=False)

    def forward(self, x):
        res_bf = self.c1_2(self.relu1(self.c1_1(x)))
        return res_bf


class CAVAR(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(CAVAR, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.c2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False)
        self.motion = Motion_var(in_planes)

    def forward(self, x):
        x_sig = self.sigmoid(self.fc2(self.relu2(self.fc1(self.avg_pool(x)))))
        x_var = self.motion(x)  # res_bf * x_var
        x_mul = x * x_sig
        out = (x + x_mul + x_var)
        return out


class Motion_var(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(Motion_var, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        mean_x = torch.mean(key, dim=1, keepdim=True)
        variance_x = torch.pow(key - mean_x, 2)
        variance_key = self.SoftMax(variance_x)
        xview = x.view(b, c, h * w).permute(0, 2, 1)
        x_mul = xview * variance_key
        x_out = x_mul.reshape(b, c, h, w).contiguous()
        return x_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(avg_out)
        return self.sigmoid(x)


class CSA(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(CSA, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 32, 1, bias=False)
        self.conv2 = nn.Conv2d(in_planes, 32, 3, padding=3, dilation=3, bias=False)
        self.conv3 = nn.Conv2d(in_planes, 32, 3, padding=5, dilation=5, bias=False)
        self.conv4 = nn.Conv2d(in_planes, 32, 3, padding=7, dilation=7, bias=False)
        self.sa = SpatialAttention(kernel_size)
        self.conv = nn.Conv2d(32 * 4, in_planes, 1, bias=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        call = torch.cat([c1, c2, c3, c4], dim=1)
        outp = self.sa(call) * call
        outp = self.conv(outp)
        return outp


class MGM_ped2(nn.Module):
    def __init__(self, t_length=3, n_channel=3):
        super(MGM_ped2, self).__init__()

        self.decoder = Decoder(t_length, n_channel)
        self.decoder_b = Decoder(t_length, n_channel)
        self.v1 = CAVAR(32)
        self.v2 = CAVAR(64)
        self.v3 = CAVAR(128)

        self.v4 = CAVAR(32)
        self.v5 = CAVAR(64)
        self.v6 = CAVAR(128)
        self.encoder = Encoder(t_length, n_channel)
        self.encoder_b = Encoder(t_length, n_channel)

    def forward(self, xf, xb):
        y_f, skf1, skf2, skf3 = self.encoder(xf)
        skf1, skf2, skf3 = self.v1(skf1), self.v2(skf2), self.v3(skf3)
        output_f = self.decoder(y_f, skf1, skf2, skf3)
        y_b, skb1, skb2, skb3 = self.encoder_b(xb)
        skb1, skb2, skb3 = self.v4(skb1), self.v5(skb2), self.v6(skb3)
        output_b = self.decoder_b(y_b, skb1, skb2, skb3)
        return {'f2b': output_f, 'b2f': output_b}
