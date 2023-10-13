import os.path

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform=transforms.Compose([
    transforms.ToTensor()
]
)


class MyDataSet(Dataset):
    def __init__(self,path):
        self.path=path
        #通过os将源路径和标签文件夹拼接生成标签目录,然后通过listdir获取所有文件名
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))


    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name=self.name[index]  #格式:xx.png
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)   #获取了标签的路径
        image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))    #获取原图路径,注意文件名不统一
        segment_image=keep_image_size_open(segment_path)
        image=keep_image_size_open(image_path)          #通过调用图像处理函数将图像的size保持一致
        return transform(segment_image),transform(image)  #通过transform将图片格式从二维数组转为张量
#print(os.listdir('./'))

# if __name__ == '__main__':
#     data=MyDataSet('D:\\PythonProjects\\Unet-Data\\VOCdevkit\\VOC2012')
#     print(data[0][0].shape)
#     print(data[0][1].shape)