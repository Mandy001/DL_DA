import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# import config as cf
def generate_data():
    x=np.zeros((0,1608))
    b_size=128
    y=np.zeros((b_size))
    y=np.hstack((y,np.zeros((b_size))+1))
    y=np.hstack((y,np.zeros((b_size))+2))

    for i in range(0,3):
        model=torch.load("netTG_{}.pt".format(i))
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


        noise = torch.randn(b_size, 100, device=device)
        sample=model(noise).cpu().detach().numpy()
        x=np.vstack((x,sample))

    return x,y
from data_loader import *
def visual(x,y):

    X,Y=return_data()
    all_X=np.vstack((x,X))
    all_Y=np.hstack((y,Y))

    X = x
    Y = y

    all_embed = TSNE(n_components=2).fit_transform(X)
    colors=['r','g','b','y','c','m']


    for i in range(0,3):
        # new_array=all_embed[Y==i]
        new_array = all_embed[Y == i]
        plt.scatter(new_array[:,0],new_array[:,1],color=colors[i],s=5)
    plt.show()

if __name__ == '__main__':
    x,y=generate_data()
    visual(x,y)
    np.savez("aug_dataT_test",x,y)
