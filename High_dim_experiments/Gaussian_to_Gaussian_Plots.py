import matplotlib
matplotlib.use('tkagg') #'tkagg) # AGG
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

num_trials = 4
num_epochs = 20

loss_traject_alpha_1 = np.zeros((num_trials, num_epochs))
loss_traject_alpha_5 = np.zeros((num_trials, num_epochs))
loss_traject_alpha_10 = np.zeros((num_trials, num_epochs))

# alpha_1 stuff
for i, trial in enumerate([1,2,3,4]):
    file = pd.read_csv('results.csv'.format(trial))\
                .to_numpy()
    loss_traject_alpha_1[i,:] = file[:num_epochs, 1]

# alpha_5 stuff
for i, trial in enumerate([6,7,8,9]):
    file = pd.read_csv('./Results_Stand_Gauss_Gauss/input_dim_784/init_trunc_inv_sqrt/layers_3/neuron_1024/lambda_cvx_0.1_mean_0.0/optim_Adamlr_0.0001betas_0.5_0.99/gen_16/batch_60/trial_{0}_last_inp_qudr/results.csv'.format(trial))\
                .to_numpy()
    loss_traject_alpha_5[i,:] = file[:num_epochs, 1]

# alpha_10 stuff
for i, trial in enumerate([11,12,13,14]):
    file = pd.read_csv('./Results_Stand_Gauss_Gauss/input_dim_784/init_trunc_inv_sqrt/layers_3/neuron_1024/lambda_cvx_0.1_mean_0.0/optim_Adamlr_0.0001betas_0.5_0.99/gen_16/batch_60/trial_{0}_last_inp_qudr/results.csv'.format(trial))\
                .to_numpy()
    loss_traject_alpha_10[i,:] = file[:num_epochs, 1]
    

loss_traject_alpha_1 /= 392
loss_traject_alpha_5 /= 392*25
loss_traject_alpha_10 /= 392*100
    
plt.figure()

# plt.plot(range(num_epochs), np.mean(loss_traject_alpha_1,0),label = r'Estimated value: $\tilde{W}_2^2(P,Q)$')
# plt.plot(range(num_epochs), 392*np.ones((num_epochs)), linestyle='--', label= r'Actual value: $W_2^2(P,Q)=392$')

# plt.plot(range(num_epochs), np.mean(loss_traject_alpha_5,0),label = r'$\alpha=5$')
# plt.plot(range(num_epochs), 392*25*np.ones((num_epochs)), linestyle='--')

plt.plot(range(num_epochs), np.mean(loss_traject_alpha_1,0),label = r'$\alpha=1$')
plt.plot(range(num_epochs), np.mean(loss_traject_alpha_5,0),label = r'$\alpha=5$')
plt.plot(range(num_epochs), np.mean(loss_traject_alpha_10,0),label = r'$\alpha=10$')

plt.plot(range(num_epochs), np.ones((num_epochs)), linestyle='--')

# plt.plot(range(num_epochs), np.mean(loss_traject_alpha_5,0),label = r'$\alpha=5$')
# plt.plot(range(num_epochs), np.mean(loss_traject_alpha_10,0),label = r'$\alpha=10$')

# plt.semilogy(test_snrs[:len(best_bers_subspace)], best_bers_full, label = 'Ours-Full Projections')

plt.xlabel('Epochs', fontsize=15)
plt.ylabel(r'$\frac{\tilde{W}_2^2(P,Q)}{W_2^2(P,Q)}$', fontsize=15, rotation=0)
xint = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
matplotlib.pyplot.xticks(xint)
# plt.title("BER plot for best model trained")
plt.grid(linestyle='--')
plt.legend(fontsize=15)
plt.savefig( '../Gaussian_Gaussian_all_alphas.pdf')
plt.show()
plt.close()

