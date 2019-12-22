import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import math

def adjust_dyn_range(x,drange_in,drange_out):
    if not drange_in==drange_out:
        scale=float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias=drange_out[0]-drange_in[0]*scale
        x=x.mul(scale).add(bias)
    return x

def resize(x,size):
    transform=transforms.Compose([transforms.ToPILImage(),
                                  transforms.Scale(size),
                                  transforms.ToTensor()])
    return transform(x)

def make_image_grid(x,ngrid):
    x=x.clone().cpu()
    if pow(ngrid,2)<x.size(0):
        grid=make_grid(x[:ngrid*ngrid],nrow=ngrid,padding=0,normalize=True,scale_each=False)
    else:
        grid=torch.Tensor(ngrid*ngrid,x.size(1),x.size(2),x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid=make_grid(grid,nrow=grid,padding=0,normalize=True,scale_each=False)
    return grid

def save_image_single(x,path,imsize=512):
    grid=make_image_grid(x,1)
    ndarr=grid.mul(255).clamp(0,255).byte().permute(1,2,0).numpy()
    im=Image.fromarray(ndarr)
    im=im.resize((imsize,imsize),Image.NEAREST)
    im.save(path)

def save_image_grid(x,path,imsize=512,ngrid=4):
    grid=make_image_grid(x,ngrid)
    ndarr=grid.mul(255).clamp(0,255).byte().permute(1,2,0).numpy()
    im=Image.fromarray(ndarr)
    im=im.resize((imsize,imsize),Image.NEAREST)
    im.save(path)

def load_model(net,path):
    net.load_state_dict(torch.load(path))

def save_model(net,path):
    torch.save(net.state_dict(),path)


def mkdir(path):
    if os.name=='nt':
        os.system('mkdir {}'.format(path.replace('/','\\')))  #勿在mkdir后遗漏空格以免创建路径失败！！！
    else:
        os.system('mkdir -r {}'.format(path))


def save_image(tensor,filename,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0):
    tensor=tensor.cpu()
    grid=make_grid(tensor,nrow=nrow,padding=padding,pad_value=pad_value,normalize=normalize,range=range,scale_each=scale_each)
    ndarr=grid.mul(255).clamp(0,255).byte().permute(1,2,0).numpy()
    im=Image.fromarray(ndarr)
    im.save(filename)


irange = range #否则'NoneType' object is not callable？！
def make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0):
    if not (torch.is_tensor(tensor)or
                (isinstance(tensor,list)and all(torch.is_tensor(t)for t in tensor))):
        raise TypeError('tensor or list of tensors expected,got{}'.format(type(tensor)))

    if isinstance(tensor,list):
        tensor=torch.stack(tensor,dim=0)

    if tensor.dim()==2:
        tensor=tensor.view(1,tensor.size(0),tensor.size(1))
    if tensor.dim()==3:
        if tensor.size(0)==1:
            tensor=torch.cat((tensor,tensor,tensor),0)
        return tensor
    if tensor.dim()==4 and tensor.size(1)==1:
        tensor=torch.cat((tensor,tensor,tensor),1)

    if normalize is True:
        tensor=tensor.clone()
        if range is not None:
            assert isinstance(range,tuple),\
                'range has to be a tuple (min, max) if specified. min and max are numbers'

        def norm_ip(img,min,max):
            img.clamp_(min=min,max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t,range):
            if range is not None:
                norm_ip(t,range[0],range[1])
            else:
                norm_ip(t,t.min(),t.max())

        if scale_each is True:
            for t in tensor:
                norm_range(t,range)
        else:
            norm_range(tensor,range)

    nmaps=tensor.size(0)
    xmaps=min(nrow,nmaps)
    ymaps=int(math.ceil(float(nmaps)/xmaps))
    height=int(tensor.size(2)+padding)
    width =int(tensor.size(3)+padding)
    grid=tensor.new(3,height*ymaps+padding,width*xmaps+padding).fill_(pad_value)
    k=0

    for y in irange(ymaps):
        for x in irange(xmaps):
            if k>=nmaps:
                break
            grid.narrow(1,y*height+padding,height-padding)\
                     .narrow(2,x*width+padding,width-padding)\
                     .copy_(tensor[k])
            k=k+1

    return grid



































