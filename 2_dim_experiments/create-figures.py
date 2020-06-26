''' the code to create the figures'''
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ot
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def main():
    create_GMM_high_dim_plot()


def create_GMM_high_dim_plot():
    data_dict = np.load("data/GMM_high_dim.npy", allow_pickle = True).item()
    X_plot = data_dict['X']
    Y_plot = data_dict['Y']
    X_pred = data_dict['X_pred']
    (N,d) = np.shape(X_plot)
    N = 1024

    import seaborn as sns

    plt.figure(figsize=(8,4))
    ax = plt.subplot(1,2,1)
    #sns.kdeplot(X_plot[:,0], X_plot[:,1], cmap="Blues", shade=True, shade_lowest=True)
    plt.scatter(X_plot[:,0], X_plot[:,1], color='C2', 
            alpha=0.15)
    plt.scatter(X_pred[:,0], X_pred[:,1], color='C3', 
            alpha=0.15)
    plt.xlabel(r'$x_1$', fontsize=15)
    plt.ylabel(r'$x_2$', fontsize=15, rotation = 0)
    plt.scatter([], [], color='C2',label='Target')
    plt.scatter([], [], color='C3', label='Transport')
    #plt.legend(loc = 3, bbox_to_anchor=(0,1.0,1.0,0.2), ncol = 2, fontsize=12)
    #plt.legend(ncol = 2, fontsize=12)
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    #plt.grid()
    

    ax.xaxis.labelpad = 0
    
    ax = plt.subplot(1,2,2)
    sns.kdeplot(X_pred[:,2], X_pred[:,3], cmap="Blues", shade=True, shade_lowest=True)
    plt.scatter(X_plot[:,2], X_plot[:,3], color='C2', 
            alpha=0.2)
    plt.scatter(X_pred[:,2], X_pred[:,3], color='C3', 
            alpha=0.1)
    plt.xlabel(r'$x_3$', fontsize=15)
    plt.ylabel(r'$x_4$', fontsize=15, rotation = 0)
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    plt.yticks([],'')
    plt.scatter([], [], color='C2',label='Target')
    plt.scatter([], [], color='C3', label='Transport')
    plt.legend(loc=2, fontsize=15)
    ax.xaxis.labelpad = 0
    ax.yaxis.set_label_coords(-0.075, 0.5)


    plt.savefig('../../GMM_high_dim.pdf')
    plt.show()


if __name__=='__main__':
    main()     

