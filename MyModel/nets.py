import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nets import ConvBlock



class Unet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super(Unet, self).__init__()
        
        self.input_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7))

        self.down1 = ConvBlock(64, 128)
        self.down2 = ConvBlock(128, 256)
        self.down3 = ConvBlock(256, 512)
                           
        self.up3 = ConvBlock(512, 256, transpose=True)
        self.up2 = ConvBlock(256, 128, transpose=True)
        self.up1 = ConvBlock(128, 64, transpose=True)

        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7))

    def forward(self, x):

        x = self.input_layer(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        x = self.output_layer(x)

        return torch.tanh(x)

class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()
        self.unet = Unet(3, 3)

    def forward(self, input):

        output = self.unet(input)

        return output