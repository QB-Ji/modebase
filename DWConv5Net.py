import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
class Convbn(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Convbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        return out 

class DWConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DWConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out 

class DWConv5Net(nn.Module):
    def __init__(self):
        super(DWConv5Net, self).__init__()
        self.feature = nn.Sequential(
            Convbn(1, 32, 2),     
            DWConv(32, 48, 1),
            DWConv(48, 64, 2),
            DWConv(64, 96, 1),
            DWConv(96, 128, 2),
            DWConv(128, 128, 2)  #(C, H/16, W/16)
        )
        self.pool = nn.AvgPool2d(kernel_size=6)
        self.classifier = nn.Linear(128, 2)
    def forward(self, x):
        out = self.feature(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out,1)
        return out 

def test():
    net = DWConv5Net()
    x = torch.randn(32, 1, 96, 96)
    x = Variable(x)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    test()
