# OT-ICNN
Code and dataset for the article [optimal transport mapping via input convex neural neworks](https://arxiv.org/abs/1908.10962)

# Optimal transport mapping via input convex neural networks

The following are the details for all the experiments reported in the paper. First we present the training details for various methods and then include the details for plotting the figures presented in the paper.

## Training the models

### 2-dimensional experiments (2_dim_experiments/)

#### Baselines (2_dim_experiments/baselines/)

```
## Checkerboard dataset

# Barycentric-OT
python main.py --solver=bary_ot --dual_iters=200000 --map_iters=200000 --data=our_checkerboard

# W2-GAN
python main.py --solver=w2 --gen=1  --train_iters=200000 --data=our_checkerboard

# W1-LP
python main.py --solver=w1 --clamp --train_iters=200000 --data=our_checkerboard


## Mixture of eight Gaussians dataset

# Barycentric-OT
python main.py --solver=bary_ot --dual_iters=200000 --map_iters=200000 --data=8gaussians

# W2-GAN
python main.py --solver=w2 --gen=1  --train_iters=200000 --data=8gaussians

# W1-LP
python main.py --solver=w1 --clamp --train_iters=200000 --data=8gaussians

```

#### Our algorithm (2_dim_experiments/)

```
## Checkerboard dataset
python W2-minimax-tf.py --DATASET_X = 'checker_board_four' --DATASET_Y='checker_board_five'

## Mixture of eight Gaussians dataset
python W2-minimax-tf.py --DATASET_X = '8gaussians' --DATASET_Y='simpleGaussian' --BATCH_SIZE = 256

```

### High dimensional experiments (High_dim_experiments/)

```
## Gaussian -> Gaussian. The parameter SCALE in the code is $\alpha$ in the paper.
python Gaussian_to_Gaussian.py --SCALE=10.0

## Gaussian -> Mixture of Gaussian.
python GMM_to_GMM.py

## MNIST {0,1,2,3,4} -> MNIST {5,6,7,8,9}.
python vaeMNIST_to_vaeMNIST.py 

## Gaussian -> MNIST.
python Gaussian_to_vaeMNIST.py 
```

## Plotting the figures in paper

### 2-dimensional experiments (2_dim_experiments/)
```
python create-2dim-figures.py
```
### High dimensional experiments (High_dim_experiments/)

```
## Figure for Gaussian -> Gaussian. 
python Gaussian_to_Gaussian_Plots.py

## Table for Gausian -> Gaussian: The relative error values for mean vectors in Table 1 are stored in the file 'relative_mean_loss_stuff.csv' for $\alpha=1, 5, 10$ over 4 different trials.

## Gaussian -> Mixture of Gaussian.
python create-GMM-figure.py

## MNIST {0,1,2,3,4} -> MNIST {5,6,7,8,9}.
python MNIST_to_MNIST_Plots.py

## Gaussian -> MNIST.
python Gaussian_to_MNIST_Plots.py
```



## Acknowledgments
```
* https://github.com/jshe/wasserstein-2
* https://github.com/mikigom/large-scale-OT-mapping-TF.git
* https://github.com/igul222/improved_wgan_training.git
```

## Dependencies

```
torch (0.4.1)
tensorflow (1.12.0)
numpy (1.14.3)
h5py (2.7.1)
torchvision (0.2.1)
scikit-learn (sklearn) (0.19.1)
matplotlib (2.2.2)
python (3.6)
tensorboardX (optional, remove dependency if not used)
```
