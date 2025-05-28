"""
Model definition taken from REPAIR. (Jordan et al. 2022)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import yaml

dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100
}

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100
}


cfg = {
    "VGG11": {
        "layers": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "num_classes": dataset_num_classes.get(config['dataset'], 10) 
    }
}

cfg = {
    "VGG11": {
        "layers": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    }
}

class VGG11_nofc(nn.Module):
    def __init__(self, vgg_name, config, w=1, init_weights=True):
        super(VGG11_nofc, self).__init__()
        self.w = w
        self.num_classes = dataset_num_classes.get(config['dataset'], 10)
        print("the number of classes", self.num_classes)
        self.features = self._make_layers(cfg[vgg_name]["layers"])
        self.classifier = nn.Linear(self.w * 512, self.num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels if in_channels == 3 else self.w * in_channels,
                        self.w * x,
                        kernel_size=3,
                        padding=1,
                    )
                )
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("Initializing with Kaiming Normal", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                print("Initializing with Normal", m)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
###############################################################################################################################################

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            """
            For CIFAR10 ResNet paper uses:
            
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            """ 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks,config, w=1):
        super(ResNet, self).__init__()

        if config['dataset'] == "cifar10":
            self.num_classes = 10
        else:
            self.num_classes = 100
        self.in_planes = w*16
        self.conv1 = nn.Conv2d(3, w*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w*16)
        self.layer1 = self._make_layer(block, w*16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w*32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w*64, num_blocks[2], stride=2)
        print("the number of classes", self.num_classes)
        self.linear = nn.Linear(w*64, self.num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
###############################################################################################################################################

def resnet50():
    model = torchvision.models.resnet50(weights=None)
    return model.cuda()

def get_blocks(net):
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                         *net.layer1, *net.layer2, *net.layer3, *net.layer4)
    