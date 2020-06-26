import os
import time
import utils
import torch
import losses
import numpy as np
import random
import glob

from torch.backends import cudnn
from data_loader import get_loader
from w1_model import W1
from w2_model import W2
from bot_model import BaryOT
from options import Options
from tensorboardX import SummaryWriter

def main():
    config = Options().parse()
    utils.print_opts(config)
    
    ## set up folders
    dir_string = './{0}_{1}/trial_{2}/'.format(config.solver, config.data, config.trial) if config.solver != 'w2' else \
                                    './{0}_gen{2}_{1}/trial_{3}/'.format(config.solver, config.data, config.gen, config.trial)

    exp_dir = dir_string #os.path.join(config.exp_dir, config.exp_name)
    model_dir = os.path.join(exp_dir, 'models')
    img_dir = os.path.join(exp_dir, 'images')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if config.use_tbx:
        # remove old tensorboardX logs
        logs = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if len(logs) > 0:
            os.remove(logs[0])
        tbx_writer = SummaryWriter(exp_dir)
    else:
        tbx_writer = None

    ## initialize data loaders & model
    r_loader, z_loader = get_loader(config)
    if config.solver == 'w1':
        model = W1(config, r_loader, z_loader)
    elif config.solver == 'w2':
        model = W2(config, r_loader, z_loader)
    elif config.solver == 'bary_ot':
        model = BaryOT(config, r_loader, z_loader)
    cudnn.benchmark = True
    networks = model.get_networks(config)
    utils.print_networks(networks)

    fixed_r, fixed_z = model.get_fixed_data()
    utils.visualize_single(utils.to_data(fixed_z), utils.to_data(fixed_r), None,
                           os.path.join(img_dir, 'data.png'), data_range = (-12,12) if config.data=='8gaussians' else (-6,6))
    if not config.no_benchmark:
        print('computing discrete-OT benchmark...')
        start_time = time.time()
        cost = model.get_cost()
        discrete_tz = utils.solve_assignment(fixed_z, fixed_r, cost, fixed_r.size(0))
        print('Done in %.4f seconds.' % (time.time() - start_time))
        utils.visualize_single(utils.to_data(fixed_z), utils.to_data(fixed_r),
                        utils.to_data(discrete_tz), os.path.join(img_dir, 'assignment.png'))

    ## training
    ## stage 1 (dual stage) of bary_ot
    start_time = time.time()
    if config.solver == 'bary_ot':
        print("Starting: dual stage for %d iters." % config.dual_iters)
        for step in range(config.dual_iters):
            model.train_diter_only(config)
            if ((step+1) % 10) == 0:
                stats = model.get_stats(config)
                end_time = time.time()
                stats['disp_time'] = (end_time - start_time) / 60.
                start_time = end_time
                utils.print_out(stats, step+1, config.dual_iters, tbx_writer)
        print("dual stage complete.")

    ## main training loop of w1 / w2 or stage 2 (map stage) of bary-ot
    map_iters = config.map_iters if config.solver == 'bary_ot' else config.train_iters
    if config.solver == 'bary_ot':
        print("Starting: map stage for %d iters." % map_iters)
    else:
        print("Starting training...")
    for step in range(map_iters):
        model.train_iter(config)
        if ((step+1) % 10) == 0:
            stats = model.get_stats(config)
            end_time = time.time()
            stats['disp_time'] = (end_time - start_time) / 60.
            start_time = end_time
            if not config.no_benchmark:
                if config.gen:
                    stats['l2_dist/discrete_T_x--G_x'] = losses.calc_l2(fixed_z, model.g(fixed_z), discrete_tz).data.item()
                else:
                    stats['l2_dist/discrete_T_x--T_x'] = losses.calc_l2(fixed_z, model.get_tx(fixed_z, reverse=True), discrete_tz).data.item()
            utils.print_out(stats, step+1, map_iters, tbx_writer)
        if ((step+1) % 10000) == 0 or step == 0:
            images = model.get_visuals(config)
            utils.visualize_iter(images, img_dir, step+1, config, data_range = (-12,12) if config.data=='8gaussians' else (-6,6))
    print("Training complete.")
    networks = model.get_networks(config)
    utils.save_networks(networks, model_dir)

if __name__ == '__main__':
    main()
