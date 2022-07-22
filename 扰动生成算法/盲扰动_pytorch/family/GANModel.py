"""
扰动生成器模型:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FamilyGeneratorModel(nn.Module):
    def __init__(self):
        super(FamilyGeneratorModel, self).__init__()
        self.layer1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.layer2=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,stride=1)
        self.layer3=nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1,stride=1)
        self.layer4=nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,padding=1,stride=1)
        self.layer5=nn.Conv2d(in_channels=4,out_channels=1,kernel_size=3,padding=1,stride=1)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        return x

if __name__=="__main__":
    model=FamilyGeneratorModel()
    inx=torch.normal(mean=0,std=1,size=(16,1,11,11))
    outx=model(inx)
    print(outx.shape)