

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import config as cf

import pandas as pd
import numpy as np
from tqdm import tqdm
import math

def MaxMinNormalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    for i in range(len(x)):
        x[i][0] = (x[i][0] - x_min)/(x_max - x_min)
    return x

def get_feature(data):
    feature = np.zeros((1608,1),dtype=np.float32)
    j = 2
    for i in range(len(data)-2):
        feature[i] = float(data.array[j])
        j += 1
    return MaxMinNormalize(feature)


def get_data(file_path):
    data_label = []
    data_x = []
    df = pd.read_excel(file_path, sheet_name='Sheet5')
    treat_label = []

    for index,row in enumerate(df.items()):
        if index >= 2:
            if math.isnan(float(row[1][2])) == True:
                continue
            value = row[1][0]
            if value == 'BCC':
                label = 0
            elif value == 'NORMAL':
                label = 1
            elif value == 'SCC':
                # else:
                label = 2
            else:
                print(label)
            data_label.append(label)
            feature = get_feature(row[1])[:, 0]
            data_x.append(feature)

            treat_label.append(row[1][1])


    data_x = np.array(data_x)
    data_label = np.array(data_label)
    return data_x,data_label, np.array(treat_label)

def return_data():
    file_path = '../data/data_feature.xlsx'
    X, Y, _ = get_data(file_path)
    return X,Y

def data_loader(label):

    file_path = '../data/data_feature.xlsx'
    X, Y, _ = get_data(file_path)
    X=X[Y==label]
    Y=Y[Y==label]
    X=np.expand_dims(X,axis=1)

    dataset=TensorDataset(torch.Tensor(X),torch.Tensor(Y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cf.batch_size,
                                             shuffle=True, num_workers=cf.workers)
    return dataloader
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and cf.ngpu > 0) else "cpu")





# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)