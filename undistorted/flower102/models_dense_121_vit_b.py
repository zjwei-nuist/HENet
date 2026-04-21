import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import timm


class Backbone(nn.Module):
    def __init__(self, generator1_arch, generator2_arch):
        super(Backbone, self).__init__()
        self.encoder1 = timm.create_model(generator1_arch, pretrained=True)
        self.encoder2 = timm.create_model(generator2_arch, pretrained=True)
        for param in self.encoder2.parameters():
            param.requires_grad = False

    def forward(self, x):
        out1 = self.encoder1.forward_features(x)
        out2 = self.encoder2.forward_features(x)
        return out1, out2


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out.expand_as(x)
        return out + x


class Decoder_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Decoder_block, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Final_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Final_block, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.Tanh()

    def forward(self, x):
        out = self.convT(x)
        out = self.act(out)
        return out


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels=3):
        super(Decoder, self).__init__()

        self.attention_block = SEBlock(in_channels=in_channels)

        self.decoder_block1 = Decoder_block(in_channels=in_channels, out_channels=in_channels // 2,
                                            kernel_size=4, stride=2, padding=1)

        self.decoder_block2 = Decoder_block(in_channels=in_channels // 2, out_channels=in_channels // 4,
                                            kernel_size=4, stride=2, padding=1)

        self.decoder_block3 = Decoder_block(in_channels=in_channels // 4, out_channels=in_channels // 8,
                                            kernel_size=4, stride=2, padding=1)

        self.decoder_block4 = Decoder_block(in_channels=in_channels // 8, out_channels=in_channels // 16,
                                            kernel_size=4, stride=2, padding=1)

        self.final_block = Final_block(in_channels=in_channels // 16, out_channels=out_channels,
                                       kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.attention_block(x)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        x = self.final_block(x)
        return x


class AdaptiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels // 2, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention_block = SEBlock(in_channels=out_channels)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.relu(self.norm3(self.conv3(out)))
        out = self.attention_block(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels // 2, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention_block = SEBlock(in_channels=out_channels)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.relu(self.norm3(self.conv3(out)))
        out = self.attention_block(out)
        out = out + x
        return out


class Fusion(nn.Module):

    def __init__(self, in_channels1, in_channels2, mid_channels):
        super(Fusion, self).__init__()

        self.adaptive_block1 = AdaptiveBlock(in_channels1, mid_channels)
        self.adaptive_block2 = AdaptiveBlock(in_channels2, mid_channels)
        self.fusion_block = FusionBlock(mid_channels, mid_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x1, x2):
        x1 = self.adaptive_block1(x1)
        x2 = self.avg_pool(x2)
        x2 = self.adaptive_block2(x2)
        out3 = self.fusion_block(x1 + x2)
        return out3


class Generator(nn.Module):
    def __init__(self, generator1_arch, generator2_arch):
        super(Generator, self).__init__()

        self.encoder = Backbone(generator1_arch, generator2_arch)
        self.in_channels1 = self.encoder.encoder1.classifier.in_features
        self.in_channels2 = self.encoder.encoder2.head.in_features
        self.mid_channels = 1024
        self.fusion = Fusion(in_channels1=self.in_channels1,
                             in_channels2=self.in_channels2,
                             mid_channels=self.mid_channels)
        self.decoder = Decoder(in_channels=self.mid_channels)

    def forward(self, x):
        x1, x2 = self.encoder(x)
        x2 = x2[:, 1:, :].reshape(x1.size(0), 14, 14, -1).permute(0, 3, 1, 2).contiguous()
        x = self.fusion(x1, x2)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output



