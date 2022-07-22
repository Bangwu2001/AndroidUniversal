"""
本地替代模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FamilyTargetModel(nn.Module):
    def __init__(self):
        super(FamilyTargetModel, self).__init__()
        self.layer1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        self.layer2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        self.layer3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        self.layer4=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1)
        self.layer5=nn.Flatten()
        self.layer6=nn.Linear(32*3*3,16)
        self.layer7=nn.Linear(16,1)
        # self.apply(self.weights_init_)
    def weights_init_(self,m):
        if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=F.relu(self.layer6(x))
        x=self.layer7(x)
        out=torch.sigmoid(x)
        return out


if __name__=="__main__":
    net=FamilyTargetModel()
    inx=torch.normal(mean=0,std=1,size=(16,1,11,11))
    outx=net(inx)
    print(outx.shape)
    print(outx)