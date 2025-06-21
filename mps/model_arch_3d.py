import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional, Type, List, Union

def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     groups=groups, dilation=dilation, bias=False)

def conv1x1_3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3_3d(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock3D]], layers: List[int],
                 num_classes: int = 4, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1

        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_3d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)   # shape: [B, 64, D/2, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # shape: [B, 64, D/4, H/4, W/4]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)   # [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)          # [B, num_classes]

        return x

if __name__ == "__main__":
    model = ResNet3D(BasicBlock3D, [3, 6, 4, 3], num_classes=4)
    dummy_input = torch.randn(2, 1, 20, 200, 200)  # [B, C, D, H, W]
    print(model(dummy_input).shape)  # Expected: [2, 4]
