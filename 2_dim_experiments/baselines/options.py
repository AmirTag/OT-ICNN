import argparse

class Options(object):
    def __init__(self):
        return

    def parse(self):
        parser = argparse.ArgumentParser()

        # experiment
        parser.add_argument('--exp_name', type=str, default='W2Experiments')
        parser.add_argument('--trial', type=int, default=1)
        parser.add_argument('--no_benchmark', action='store_true', help='do not compare against discrete OT benchmark')
        parser.add_argument('--exp_dir', type=str, default='.')
        parser.add_argument('--use_tbx', type=int, default=1, help='use tensorboardX')
        parser.add_argument('--solver', type=str, choices=['w2', 'w1', 'bary_ot'], default='w2')
        parser.add_argument('--gen', type=int, default=1, choices=[0, 1], help='represent map with a neural network (ex. GAN generator) as opposed to closed form expression; required for bary-OT')

        # training
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--d_lr', type=float, default=0.0005)
        parser.add_argument('--g_lr', type=float, default=0.0001)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        # w1 / w2 only
        parser.add_argument('--train_iters', type=int, default=5000)
        parser.add_argument('--d_iters', type=int, default=5, help='# d updates per g update; defaults to 1 if solver=bary_ot')
        # bary-ot only (2 stages)
        parser.add_argument('--dual_iters', type=int, default=20000)
        parser.add_argument('--map_iters', type=int, default=20000)

        # data
        parser.add_argument('--data', type=str, choices=['4gaussians', 'swissroll', 'checkerboard', '8gaussians', 'our_checkerboard'], default='our_checkerboard')

        # networks
        parser.add_argument('--g_n_layers', type=int, default=3)
        parser.add_argument('--d_n_layers', type=int, default=3)
        parser.add_argument('--n_hidden', type=int, default=128)
        parser.add_argument('--g_norm', type=str, choices=['none', 'batch'], default='batch', help='normalization (generator only)')
        parser.add_argument('--activation', type=str, choices=['relu', 'elu'], default='relu', help='activation function')

        # losses
        # w2 only
        parser.add_argument('--ineq', type=int, default=1, choices=[0, 1], help='inequality regularization for (x, y)')
        parser.add_argument('--ineq_interp', type=int, default=1, choices=[0, 1], help='inequality regularization for interpolations between x and y')
        parser.add_argument('--eq_phi', type=int, default=1, choices=[0, 1], help='equality regularization for (x, Tx)')
        parser.add_argument('--eq_psi', type=int, default=-1, choices=[-1, 0, 1], help='equality regularization for (y, Ty); -1 defaults to eq_phi')
        parser.add_argument('--lambda_ineq', type=float, default=200, help='weight of ineq regularizer')
        parser.add_argument('--lambda_eq', type=float, default=-1, help='weight of eq regularizer; < 0 defaults to lambda_ineq')
        parser.add_argument('--p', type=float, default=2, help='power in cost function')
        parser.add_argument('--l', type=int, default=2, help='norm of cost function')
        # w1 only
        parser.add_argument('--lambda_gp', type=float, default=10, help='weight of gradient penalty regularizer')
        parser.add_argument('--clamp', action='store_true', help='modify wgan-gp objective -> wgan-lp objective via clamping gradient')
        # bary-ot only
        parser.add_argument('--reg_type', type=str, choices=['l2', 'entropy'], default='l2')

        config = parser.parse_args()
        config.d_iters = 1 if (config.solver == 'bary_ot' or not config.gen) else config.d_iters
        config.gen = True if config.solver == 'bary_ot' else config.gen
        config.eq_psi = config.eq_phi if config.eq_psi < 0 else config.eq_psi
        config.lambda_eq = config.lambda_ineq if config.lambda_eq < 0 else config.lambda_eq
        return config
