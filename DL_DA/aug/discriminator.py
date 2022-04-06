
import torch.nn as nn

import config as cf

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu


        self.main=nn.Sequential(
            nn.Linear(cf.image_size,cf.ndf),
            # nn.BatchNorm1d(cf.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cf.ndf,cf.ndf),
            # nn.BatchNorm1d(cf.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cf.ndf,cf.ndf),
            # nn.BatchNorm1d(cf.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cf.ndf,1),
            nn.Sigmoid()
        )




    def forward(self, input):
        return self.main(input)