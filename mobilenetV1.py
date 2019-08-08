import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
class Block(nn.Module):
    """
    DWConv + PWConv
    """
    def __init__(self, in_planes, out_planes, stride):
        super(Block,self).__init__()
        """
        Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        """
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes,kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out 

class MobileNetV1(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    def __init__(self, input_channel, num_classes):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes) # 这个1024 可以再优化一下

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out 

def test():
    x = torch.randn([16, 3, 28, 28])
    net = MobileNetV1(3, 10)
    y= net(Variable(x))
    print(y.size())


test()