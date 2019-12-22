import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import copy
from torch.nn.init import kaiming_normal_, xavier_normal_, calculate_gain

class ConcatTable(nn.Module):
    def __init__(self,layer1,layer2):
        super(ConcatTable,self).__init__()
        self.layer1=layer1
        self.layer2=layer2

    def forward(self,x):
        y=[self.layer1(x),self.layer2(x)]
        return y

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)

class fadein_layer(nn.Module):
    def __init__(self,config):
        super(fadein_layer,self).__init__()
        self.alpha=0.0

    def update_alpha(self,delta):
        self.alpha=self.alpha+delta
        self.alpha=max(0,min(self.alpha,1.0))  #令alpha取值在0~1之间

    def forward(self,x):
        return torch.add(x[0].mul(1.0-self.alpha),x[1].mul(self.alpha))

class minibatch_std_concat_layer(nn.Module):
    def __init__(self,averaging='all'):
        super(minibatch_std_concat_layer,self).__init__()
        self.averaging=averaging.lower()
        if 'group' in self.averaging:
            self.n=int(self.averaging[5:])    # group 参数
        else:
            assert self.averaging in ['all','spatial','none','gpool'], 'invalid averaging mode ' %self.averaging
        self.adjusted_std=lambda x,**kwargs:torch.sqrt(torch.mean((x-torch.mean(x,**kwargs))**2,**kwargs)+1e-8)

    def forward(self,x):
        shape=list(x.size())
        target_shape=copy.deepcopy(shape)     #新版python中，deepcopy与copy无异
        vals=self.adjusted_std(x,dim=0,keepdim=True)   #样本间的标准差，余下CHW三个维度：**kwargs为dim=0,keepdim=True
        if self.averaging=='all': #再另行通道平均，余下两个维度HW
            target_shape[1]=1
            vals=torch.mean(vals,dim=1,keepdim=True)
            #vals=np.mean(vals,axis=1,keepdims=True)
        elif self.averaging=='spatial': #再另行空间平均，余下一个维度C
            if len(shape)==4:
                #vals = np.mean(vals, axis=[2, 3], keepdims=True)
                vals=torch.mean(torch.mean(vals, dim=2, keepdim=True), dim=3, keepdim=True)
        elif self.averaging=='none': #不再使用任何其他平均，方便后恢复batch_size故加此举
            target_shape=[target_shape[0]]+[s for s in target_shape[1:]]
        elif self.averaging=='gpool': #同时通道平均和空间平均，结果维度为1
            if len(shape)==4:
                #vals = np.mean(x, [0, 2, 3], keepdims=True)
                vals=torch.mean(torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True), dim=0, keepdim=True)
        else:       # self.averaging == 'group'
            target_shape[1]=self.n   #group batch size
            vals=vals.view(self.n,self.shape[1]/self.n,self.shape[2],self.shape[3])
            vals=torch.mean(vals, dim=0, keepdim=True)
            vals=vals.view(1,self.n,1,1)   # 通道间分组平均
        vals=vals.expand(*target_shape) #改变维度
        return torch.cat([x,vals],1)  # 每个样本增加一个通道

    def __repr__(self):
        return self.__class__.__name__+'(averaging=%s)'%(self.averaging)

class pixelwise_norm_layer(nn.Module):
    def __init__(self):
        super(pixelwise_norm_layer,self).__init__()
        self.eps=1e-8
    # 通道归一化
    def forward(self,x):
        return x/(torch.mean(x**2,dim=1,keepdim=True)+self.eps)**0.5

class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_conv2d,self).__init__()
        self.conv=nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True)  #偏置影响不大
        if initializer=='kaiming':
            torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv2d'))
            #  a is the negative slope of the rectifier used after this layer
        elif initializer=='xavier':
            torch.nn.init.xavier_normal_(self.conv.weight)
        #conv_w=self.conv.weight.data.clone()
        #self.bias=torch.nn.Parameter(torch.Tensor(c_out).fill_(0))
        self.scale=(torch.mean(self.conv.weight.data**2))**0.5    #将scale分散至权重及输入中加速收敛
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)  #把权重训练词向量copy进去#权重归一化

    def forward(self,x):
        x=self.conv(x.mul(self.scale))
        return x  # x+self.bias.view(1,-1,1,1).expand_as(x) # 数据增大，权重归一，不影响最终结果却优化了GAN收敛

class equalized_deconv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_deconv2d,self).__init__()
        self.deconv=nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer=='kaiming':
            torch.nn.init.kaiming_normal_(self.deconv.weight, a=calculate_gain('conv2d'))
        elif initializer=='xavier':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        #deconv_w=self.deconv.weight.data.clone()
        self.bias=torch.nn.Parameter(torch.Tensor(c_out).fill_(0))  # add a bias channel
        self.scale=(torch.mean(self.deconv.weight.data**2))**0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data/self.scale)

    def forward(self,x):
        x=self.deconv(x.mul(self.scale))
        return x+self.bias.view(1,-1,1,1).expand_as(x)

class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear=nn.Linear(c_in, c_out, bias=False)
        if initializer=='kaiming':
            torch.nn.init.kaiming_normal_(self.linear.weight, a=calculate_gain('conv2d'))
        elif initializer=='xavier':
            torch.nn.init.xavier_normal_(self.linear.weight)
        #linear_w=self.linear.weight.data.clone()
        self.bias=torch.nn.Parameter(torch.Tensor(c_out).fill_(0))
        self.scale=(torch.mean(self.linear.weight.data**2))**0.5
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1, -1).expand_as(x)

class generalized_drop_out(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(generalized_drop_out, self).__init__()
        self.mode=mode.lower()
        assert self.mode in ['mul','drop','prop'], 'invalid GProplayer mode '%mode
        self.strength=strength
        self.axes=[axes] if isinstance(axes, int) else list(axes)
        self.normalize=normalize
        self.gain=None

    def forward(self,x,deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape=[s if axis in self.axes else 1 for axis,s in enumerate(x.size())]
        if self.mode=='drop':
            p=1-self.strength
            rnd=np.random.binomial(1,p=p,size=rnd_shape)/p
        elif self.mode=='mul':
            rnd=(1+self.strength)**np.random.normal(size=rnd_shape)
        else:
            coef=self.strength*x.size(1)**0.5
            rnd=np.random.normal(size=rnd_shape)*coef+1

        if self.normalize:
            rnd=rnd/np.linalg.norm(rnd,keepdims=True)
        rnd=Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd=rnd.cuda()
        return x*rnd

    def __repr__(self):
        param_str='(mode=%s,strength=%s,axes=%s,normalize=%s)'%(self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str




















