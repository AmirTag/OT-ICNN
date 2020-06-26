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

    create_scatter_plot_for_comparison('bary_ot','checkerboard','1')
    create_scatter_plot_for_comparison('bary_ot','8gaussians','1')

    create_scatter_plot_for_comparison('w1','checkerboard','1')
    create_scatter_plot_for_comparison('w1','8gaussians','1')

    create_scatter_plot_for_comparison('w2','checkerboard','1')
    create_scatter_plot_for_comparison('w2','8gaussians','1')

    create_scatter_plot_for_comparison('minimax','checkerboard','1')
    create_scatter_plot_for_comparison('minimax','8gaussians','1')


def create_scatter_plot_for_comparison(model,dataset,trial):


    file_name = 'data/{0}_{1}_{2}.npy'.format(model,dataset,trial)
    # loading the data
    data_dict = np.load(file_name, allow_pickle = True).item()
    X_plot = data_dict['X']
    Y_plot = data_dict['Y']
    X_pred = data_dict['X_pred']
    (N,d) = np.shape(X_plot)
    N = 512

    plt.figure() 
    plt.scatter(Y_plot[:N,0], Y_plot[:N,1], color='C1', alpha=0.4)
    plt.scatter(X_plot[:N,0], X_plot[:N,1], color='C2', alpha=0.4)
    plt.scatter(X_pred[:N,0], X_pred[:N,1], color='C3', alpha=0.4)
    
    plt.scatter([],[], color='C1', label = 'Source')
    plt.scatter([],[], color='C2', label = 'Target')
    plt.scatter([],[], color='C3', label= 'Transp.')

    plt.xticks([],'')
    plt.yticks([],'')

    for i in range(N):
        drawArrow(Y_plot[i,:], X_pred[i,:])

    plt.legend(loc=3, bbox_to_anchor=(-0.015, 0.99, 1.0, 1.2), ncol=3, fontsize=15)

    plt.savefig('../../{0}_{1}_{2}.pdf'.format(model,dataset,trial))
    plt.show()


def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], color='C0', alpha = 0.1, head_width=0.001, width = 0.001)

if __name__=='__main__':
    main()     

