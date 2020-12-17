import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthNet(nn.Module):
    def __init__(self, config):
        super(DepthNet, self).__init__()
        self.config = config

        self.encoder = models.resnet18(pretrained=True)
        
        # channels in final output of resnet18 encoder
        enc_out_ch = self.encoder.layer4[-1].conv2.out_channels
        decoder_layers = []
        out_layers = []
        
        for i in range(3):
            ch = enc_out_ch // (2**i)
            decoder_layers.append(nn.Conv2d(ch, ch//2, kernel_size=3, padding=1))
            decoder_layers.append(nn.Conv2d(ch, ch//2, kernel_size=3, padding=1))
            if i > 0:
                out_layers.append(nn.Sequential(nn.Conv2d(ch//2, 1, kernel_size=3, padding=1), nn.Sigmoid()))
        
        decoder_layers += [
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)]
        out_layers += [
            nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.Sigmoid()),
            nn.Sequential(nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Sigmoid())]

        self.convs = nn.ModuleList(decoder_layers)
        self.out_convs = nn.ModuleList(out_layers)

    # img: [B,3,H,W]
    def forward(self, img):
        enc_outputs = []
        x = self.encoder.conv1(img)
        x = self.encoder.bn1(x)
        enc_outputs.append(self.encoder.relu(x))
        enc_outputs.append(self.encoder.layer1(self.encoder.maxpool(enc_outputs[-1])))
        enc_outputs.append(self.encoder.layer2(enc_outputs[-1]))
        enc_outputs.append(self.encoder.layer3(enc_outputs[-1]))
        x = self.encoder.layer4(enc_outputs[-1])

        outputs = []
        for i in range(5):
            x = self.convs[2*i](x)
            # upsample
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            # skip connection
            if i != 4:
                x = torch.cat([x, enc_outputs[-(i+1)]], dim=1)
            x = self.convs[2*i+1](x)
            # output at different scales
            if i > 0:
                outputs.append(self.out_convs[i-1](x))

        return outputs

