import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def __make_divisible(v, divisor, min_value=None):
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
    return 

class LinearBottleneck(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1, t=6, atctivate=nn.ReLU6):
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
    def __init__(self, scale=1.0, input_size=334, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
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
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        #assert(input_size % 32 == 0)

        self.c = []