import torch
import torch.nn as nn
import torchvision.models as models


class PoseNet(nn.Module):
    def __init__(self, config):
        super(PoseNet, self).__init__()
        self.config = config

        self.resnet = models.resnet18(pretrained=True)
        
        # modify input conv for 2 concatenated frames
        new_input_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_input_conv.weight.data = torch.cat([self.resnet.conv1.weight] * 2, dim=1) / 2
        self.resnet.conv1 = new_input_conv

        ch = self.resnet.layer4[-1].conv2.out_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(ch, ch//2, kernel_size=1), nn.ReLU(),
            nn.Conv2d(ch//2, ch//2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(ch//2, ch//2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(ch//2, 6, kernel_size=1)
        )

    # img_pair: [B,6,H,W]
    def forward(self, img_pair):
        # forward through resnet
        x = self.resnet.conv1(img_pair)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # mean over spatial dims
        out = self.decoder(x).mean(dim=-1).mean(dim=-1)
        return out * 0.01

