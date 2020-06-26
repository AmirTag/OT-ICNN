# Adversarial Computation of Optimal Transport Maps

Code accompanying paper of the same name.

## Dependencies

```
torch (0.4.1)
numpy (1.14.3)
h5py (2.7.1)
torchvision (0.2.1)
scikit-learn (sklearn) (0.19.1)
matplotlib (2.2.2)
python (3.6)
tensorboardX (optional, remove dependency if not used)
```

## Experiments

### 2D/OT (exp_2d)

```
# W2-GAN
# 4 gaussians
python main.py --solver=w2 --gen=1 --data=4gaussians
# swissroll
python main.py --solver=w2 --gen=1 --data=swissroll
# checkerboard
python main.py --solver=w2 --gen=1 --data=checkerboard

# W2-OT
# 4 gaussians
python main.py --solver=w2 --gen=0 --data=4gaussians --train_iters=20000
# swissroll
python main.py --solver=w2 --gen=0 --data=swissroll --train_iters=20000
# checkerboard
python main.py --solver=w2 --gen=0 --data=checkerboard --train_iters=20000
```

```

### Multivariate Gaussian âŸ¶ MNIST (exp_mvg)

```
python main.py --solver=w2
```



## Acknowledgments

* https://github.com/mikigom/large-scale-OT-mapping-TF.git
* https://github.com/igul222/improved_wgan_training.git

