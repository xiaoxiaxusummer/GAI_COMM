#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, torch, argparse
from matplotlib import pyplot as plt
import scipy
import numpy as np

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train', type=str, default='mixed')
    parser.add_argument('--test', type=str, default='mixed')
    parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
    parser.add_argument('--pilot_alpha', nargs='+', type=float, default=0.6)
    args = parser.parse_args()

    # Disable TF32 due to potential precision issues
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


    ########################### for DM / OMP Lasso/ l1CS_lifted4 ###########################
    # result_dir = f'./results/l1CS_lifted4/train-{args.train}_test-{args.test}'
    # result_dir = f'./results/DM/train-{args.train}_test-{args.test}'
    result_dir = f'./results/DM/train-{args.train}_test-{args.test}'
    #
    save_dict = torch.load( os.path.join(result_dir, 'results.pt'))
    best_nmse_data = 10 * np.log10(save_dict["best_nmse"].squeeze())

    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 10))
    plt.plot(save_dict["snr_range"], best_nmse_data, linewidth=4)
    plt.grid()
    plt.legend()
    plt.title('Channel estimation')
    plt.xlabel(f'SNR [dB] in {args.test}'); plt.ylabel('Normalized MSE [dB]')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'results_nmse.png'), dpi=300,
                bbox_inches='tight')
    plt.close()
    mse_data = save_dict["best_nmse"].squeeze()


    ############################### for ML / WGAN_GP  ################################

    # result_dir = f'./results/ml_baseline/train{args.train}_test{args.test}'
    # save_dict = torch.load( os.path.join(result_dir, 'results_Nt64_Nr16.pt'))

    # result_dir = f'./results/wgan_gp/train{args.train}_test{args.test}'
    # save_dict = torch.load( os.path.join(result_dir, 'results.pt'))

    # plt.rcParams['font.size'] = 14
    # plt.figure(figsize=(10, 10))
    # plt.plot(save_dict["snr_range"], save_dict["avg_nmse"].squeeze(), linewidth=4)
    # plt.grid(); plt.legend()
    # plt.title('Channel estimation')
    # plt.xlabel(f'SNR [dB] in {args.test}'); plt.ylabel('Normalized MSE [dB]')
    # plt.tight_layout()
    # plt.savefig(os.path.join(result_dir, 'results_nmse.png'), dpi=300,
    #             bbox_inches='tight')
    # plt.close()
    # best_nmse_data = 10*np.log10(save_dict["avg_nmse"].squeeze())
    # mse_data = save_dict["avg_nmse"].squeeze()


    ########################### save mat file  ###########################
    scipy.io.savemat(os.path.join(result_dir, 'results_mat.mat'),{'nmse':best_nmse_data, 'mse':mse_data})
