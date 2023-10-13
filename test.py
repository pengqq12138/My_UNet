from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

net=UNet().cuda()
weight_path='./params/unet.pth'
save_image_path='./test_image'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successfully load the model!')

input_image_path=input('please input the image path:')
img=keep_image_size_open(input_image_path)
img=transform(img)
img=torch.unsqueeze(img,dim=0).cuda()
print(img.size())
out=net(img)
#print(out)
save_image(out,f'{save_image_path}/result1.jpg')

# net=UNet()
# weight_path='./params/unet.pth'
# save_image_path='.'
# net.load_state_dict(torch.load(weight_path))
# img=keep_image_size_open('test1.jpg')
# img=transform(img).unsqueeze(0)
# print(f'testimg.size={img.size()}')  #(3,256,256)
# output=net(img)
# print(output.size())
# img=torch.stack([output[0],img[0]])
# save_image(img,f'{save_image_path}/test.png')

