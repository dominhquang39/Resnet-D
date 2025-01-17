"""
PyTorch implementation of ResNet-18, 34, 50, 101, 152 & ResNet-D-50, 101, 152

model names: resnet18, resnet34, resnet50, resnet101, resnet152, resnet50D, resnet101D, resnet152D

ResNet Parameters(**kwargs)
:in_channels (int, default=3): input channel dimension
:num_classes (int, default=1000): number of class labels
:zero_init_residual (bool, default=False): Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, and each residual block behaves like an identity.
:groups (int, default=1): number of groups for conv3x3
:width_per_group (int, default=64): to change model width
:replace_stride_with_dilation (list of bools, default=None): each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead
:norm_layer (nn.Module, default=None): norm layer to use

"""

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Path A
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Path B
        if self.downsample is not None:
            identity = self.downsample(x)

        # Path A + Path B
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()

        #parameters
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            # Standard striding
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        #input stem
        self.input_stem = self._make_input_stem(in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #stages 1~4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        #output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self._make_downsample(planes, block, stride, norm_layer)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _make_downsample(self, planes, block, stride, norm_layer):
        return nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    def _make_input_stem(self, in_channels):
        norm_layer = self.norm_layer
        return nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Input stem
        x = self.input_stem(x)
        x = self.maxpool(x)

        # Stages 1~4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNetD(ResNet):
    def _make_downsample(self, planes, block, stride, norm_layer): #conv1x1 -> AvgPool+conv1x1
        if stride != 1:
            return nn.Sequential(
                nn.AvgPool2d(2, stride=stride, ceil_mode=True),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        else:
            return nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

    def _make_input_stem(self, in_channels): #conv7x7 -> 3 conv3x3
        norm_layer = self.norm_layer
        return nn.Sequential( 
            nn.Conv2d(in_channels, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(self.inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3, padding=1, bias=False),
            norm_layer(self.inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, padding=1, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
        )

#ResNet
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

#ResNet-D
def resnet50D(**kwargs):
    return ResNetD(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101D(**kwargs):
    return ResNetD(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152D(**kwargs):
    return ResNetD(Bottleneck, [3, 8, 36, 3], **kwargs)