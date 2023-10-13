from torch import nn
from torch.nn import functional as F
import torch


#UNET中的卷积层,将输入通道数增加
class Conv_Block(nn.Module):
    def __init__(self,in_chennels,out_chennels):
        super(Conv_Block,self).__init__()
        self.layer=nn.Sequential(
            #padding_mode='reflect'表示翻转填充.在卷积操作之前，图像的边缘会被复制并翻转，以填充周围的边缘
            #相比于零填充,这样能保证整张图都是有特征的,便于特征提取
            nn.Conv2d(in_chennels,out_chennels,3,stride=1,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_chennels),  #对数据进行归一化处理,防止数据过大导致网络性能不稳定
            nn.Dropout2d(0.3), #防止神经网络过拟合,随机的将每个通道的输入单元以0.3的概率归零
            nn.LeakyReLU(),
            nn.Conv2d(out_chennels, out_chennels, 3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_chennels),  # 对数据进行归一化处理,防止数据过大导致网络性能不稳定
            nn.Dropout2d(0.3),  # 防止神经网络过拟合,随机的将每个通道的输入单元以0.3的概率归零
            nn.LeakyReLU(),
        )

    def forward(self,x):
        return self.layer(x)

#UNET中的下采样
class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer=nn.Sequential(
            #stride=2,使得输出的图像尺寸变为原来的一半
            nn.Conv2d(channel,channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )

    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,kernel_size=1,stride=1)



    def forward(self,x,feature_map): #由于要进行拼接,还需要拿到之前的特征图
        up=F.interpolate(x,scale_factor=2,mode='nearest')  #使用插值法对输入进行上采样,将尺寸扩大到原来的二倍
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)  #(N,C,H,W)

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,3,3,1,1)
        self.Th=nn.Sigmoid()

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2=self.c7(self.u2(O1,R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)  #N,C,W,H
    net=UNet()
    print(net(x).size())