import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block with two convolutional paths and one shortcut connection"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolution: may change image size based on stride, channels may change
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second convolution: image size unchanged, channels unchanged
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection: if image dimensions change (stride != 1) or channels change,
        # use 1x1 convolution to match dimensions for addition
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First conv + BN + ReLU
        out = self.bn2(self.conv2(out))        # Second conv + BN
        out += self.shortcut(x)                # Add shortcut connection
        out = F.relu(out)                      # Final ReLU activation
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block with three convolutional paths (1x1, 3x3, 1x1)"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # First 1x1 conv: reduces channels
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second 3x3 conv: main feature extraction, may downsample
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Third 1x1 conv: expands channels by expansion factor (4x)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First 1x1 conv + BN + ReLU
        out = F.relu(self.bn2(self.conv2(out)))  # Second 3x3 conv + BN + ReLU
        out = self.bn3(self.conv3(out))         # Third 1x1 conv + BN
        out += self.shortcut(x)                  # Add shortcut connection
        out = F.relu(out)                        # Final ReLU activation
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR datasets"""
    
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution and batch normalization: image size unchanged, channels 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Four residual layers with increasing channels and potential downsampling
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 64 channels
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 128 channels, downsample
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 256 channels, downsample
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 512 channels, downsample

        # Final classification layer
        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        # First block may have stride > 1, remaining blocks have stride = 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Update input channels for next block
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial conv + BN + ReLU
        out = self.layer1(out)                 # Layer 1
        out = self.layer2(out)                 # Layer 2
        out = self.layer3(out)                 # Layer 3
        out = self.layer4(out)                 # Layer 4
        out = F.avg_pool2d(out, 4)             # Average pooling 4x4
        out = out.view(out.size(0), -1)        # Flatten
        out = self.linear(out)                 # Classification layer
        return out


def customized_resnet18(pretrained: bool = False, class_num=10, progress: bool = True) -> ResNet:
    """ResNet-18 with GroupNorm instead of BatchNorm for federated learning"""
    res18 = ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)

    # Replace BatchNorm with GroupNorm for better federated learning performance
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18


def original_resnet18(pretrained: bool = False, class_num=10, progress: bool = True) -> ResNet:
    """Standard ResNet-18 with BatchNorm"""
    res18 = ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)
    return res18


class tiny_ResNet(nn.Module):
    """Tiny ResNet variant with adaptive average pooling"""
    
    def __init__(self, block, num_blocks, class_num=10):
        super(tiny_ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution and batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Four residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Adaptive average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial conv + BN + ReLU
        out = self.layer1(out)                 # Layer 1
        out = self.layer2(out)                 # Layer 2
        out = self.layer3(out)                 # Layer 3
        out = self.layer4(out)                 # Layer 4
        out = self.avgpool(out)                # Adaptive average pooling
        out = torch.flatten(out, 1)            # Flatten
        out = self.linear(out)                 # Classification layer
        return out


def tiny_resnet18(pretrained: bool = False, class_num=10, progress: bool = True) -> tiny_ResNet:
    """Tiny ResNet-18 with GroupNorm instead of BatchNorm"""
    res18 = tiny_ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)

    # Replace BatchNorm with GroupNorm
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18


def _replace_bn_with_gn(model, num_groups=32):
    """
    Helper function to replace all BatchNorm layers with GroupNorm.
    This is used for federated learning where BatchNorm can cause issues.
    """
    # First, collect all BatchNorm layers with their parent info
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            groups = min(num_groups, num_channels)
            gn = nn.GroupNorm(groups, num_channels)
            
            # Get parent module and child name
            name_parts = name.split('.')
            if len(name_parts) > 1:
                parent_name = '.'.join(name_parts[:-1])
                child_name = name_parts[-1]
                # Get parent module
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
                child_name = name_parts[0]
            
            bn_layers.append((parent, child_name, gn))
    
    # Now replace all BatchNorm layers
    for parent, child_name, gn in bn_layers:
        setattr(parent, child_name, gn)


def customized_resnet50(pretrained: bool = False, class_num=10, progress: bool = True) -> ResNet:
    """
    ResNet-50 adapted for CIFAR datasets (32x32 input) with GroupNorm instead of BatchNorm.
    ResNet-50 uses Bottleneck blocks with [3, 4, 6, 3] blocks per layer.
    """
    # ResNet-50 architecture: [3, 4, 6, 3] bottleneck blocks
    res50 = ResNet(Bottleneck, [3, 4, 6, 3], class_num=class_num)
    
    # Replace all BatchNorm layers with GroupNorm for federated learning
    _replace_bn_with_gn(res50, num_groups=32)
    
    # Verify all BatchNorm layers were replaced
    assert len(dict(res50.named_parameters()).keys()) == len(
        res50.state_dict().keys()), 'More BN layers are there...'
    
    return res50