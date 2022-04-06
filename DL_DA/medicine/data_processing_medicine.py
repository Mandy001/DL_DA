# coding=utf-8
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



