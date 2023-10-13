#在Python项目中，一个名为 utils.py 的文件通常用来包含各种辅助函数、工具函数或实用性函数。这些函数可能被项目中的多个模块或脚本共享，以提供通用的功能和功能。

from PIL import Image


#将图片等比缩放,最终形状为(256,256)
def keep_image_size_open(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size)      #获取图片的最长边,以构建正方形
    mask=Image.new('RGB',(temp,temp),(0,0,0))  #构建一个三通道,边长为图片最长边的正方形图片,背景为黑色
    mask.paste(img,(0,0))  #将图片粘贴到正方形的左上角
    mask=mask.resize(size)     #通过将正方形进行缩放,保证图片也被等比缩放
    return mask


# IMG=Image.open('../Unet-Data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')
# print(IMG)
# print(max(IMG.size))
# mask=Image.new('RGB',(max(IMG.size),max(IMG.size)),(0,0,0))
#
# mask.paste(IMG,(0,0))
# mask=mask.resize((256,256))
# print(mask)
test=Image.open('test_image/test1.jpg')
print(test)