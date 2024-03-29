from collections import OrderedDict 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import init 

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor 
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LinearBottleneck(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1, t=6, activate=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes*t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes*t)
        self.conv2 = nn.Conv2d(in_planes*t, in_planes*t, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_planes*t)
        self.bn2 = nn.BatchNorm2d(in_planes*t)
        self.conv3 = nn.Conv2d(in_planes*t, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.activate = activate(inplace=True)
        self.stride = stride 
        self.t = t 
        self.in_planes = in_planes
        self.out_planes = out_planes

    def forward(self, x):
        residual = x 

        out = self.activate(self.bn1(self.conv1(x)))
        out = self.activate(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.stride == 1 and self.in_planes == self.out_planes:
            out += residual
        return out 


class MobileNetV2(nn.Module):
    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """
        super(MobileNetV2, self).__init__()

        self.scale = scale 
        self.t = t 
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        #assert(input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels] #channels list 
        self.n = [1, 1, 2, 3, 4, 3, 3, 1] # how many linearBottleneck in Bottlenecks
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]  # strides list
        
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks =  self._make_bottlenecks()

        #Last convolution has 1280 output channels for scale <=1 
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8) 
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True) #
        self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = 'Bottlenecks'

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(in_planes=self.c[0], out_planes=self.c[1], n=self.n[1], stride=self.s[1], t=1, stage=0)
        modules[stage_name + "_0"] = bottleneck1 
        
        # add more linearBottleneck depending on number of repeats 
        for i in range(1, len(self.c) -1):
            name = stage_name + '_{}'.format(i)
            module = self._make_stage(in_planes=self.c[i], out_planes=self.c[i+1], n=self.n[i+1], stride=self.s[i+1], t=self.t, stage=i)
            modules[name] = module 

        return nn.Sequential(modules)

    def _make_stage(self, in_planes, out_planes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name= 'LinearBottleneck{}'.format(stage)

        #First module is the only one utilizing stride 
        first_module = LinearBottleneck(in_planes=in_planes, out_planes=out_planes, stride=stride, t=t,activate=self.activation_type)
        modules[stage_name + '_0'] = first_module

        #add more linearBottleneck depending on number or repeats 
        for i in range(n-1):
            name = stage_name + "_{}".format(i+1)
            module = LinearBottleneck(in_planes=out_planes, out_planes=out_planes, stride=1, t=6, activate=self.activation_type)
            modules[name] = module 
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer 
        x = self.avgpool(x)
        x = self.dropout(x)

        #flatten for input to fully-connected layer 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 
    
if __name__=="__main__":
    # model1 = MobileNetV2()
    # print(model1)

    # model2 = MobileNetV2(scale=0.35) 
    # print(model2)

    #model3 = MobileNetV2(in_channels=2, num_classes=10)
    # print(model3)

    # x = torch.randn(1, 2, 224, 224)
    # print(model3(x))

    # model4_size = 32 * 10
    # model4 = MobileNetV2(input_size=model4_size, num_classes=10)
    # print(model4)

    # x2 = torch.randn(1, 3, model4_size, model4_size)
    # print(model4(x2))

    model5 = MobileNetV2(input_size=196, num_classes=10)
    x3 = torch.randn(1, 3, 196, 196)
    print(model5(x3))  # fail