

import argparse
import os
import random
import torch
import torch.nn as nn
import config as cf
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main=nn.Sequential(
            nn.Linear(cf.nz,cf.ngf),
            nn.BatchNorm1d(cf.ngf),
            nn.ReLU(True),
            nn.Linear(cf.ngf,cf.ngf),
            nn.BatchNorm1d(cf.ngf),
            nn.ReLU(True),
            nn.Linear(cf.ngf,cf.ngf),
            nn.BatchNorm1d(cf.ngf),
            nn.ReLU(True),
            nn.Linear(cf.ngf,cf.image_size),
            nn.BatchNorm1d(cf.image_size),
            nn.ReLU(True),

        )



    def forward(self, input):
        return self.main(input)