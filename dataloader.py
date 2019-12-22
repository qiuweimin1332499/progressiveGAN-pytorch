import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

class dataloader:
    def __init__(self,config):
        self.root=config.train_data_root
        self.batch_table={4:32,8:32,16:32,32:16,64:16,128:16,256:12,512:3,1024:1}
        self.batch_size=int(self.batch_table[pow(2,2)])    # we start from 2^2=4
        self.imsize=int(pow(2,2))
        self.num_workers=1  #this computer only 1 GPU available!!!!!

    def renew(self,resl):
        print('[*]Renew dataloader configuration,load data from{}.'.format(self.root))
        self.batch_size=int(self.batch_table[pow(2,resl)])
        self.imsize=int(pow(2,resl))
        self.dataset=ImageFolder(root=self.root,
                                 transform=transforms.Compose([transforms.Resize(size=(self.imsize,self.imsize),interpolation=Image.NEAREST),
                                                               transforms.ToTensor(),]))
        self.dataloader=DataLoader(dataset=self.dataset,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   num_workers=self.num_workers)

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter=iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)        # pixel range [-1, 1] reformed from [0, 1]