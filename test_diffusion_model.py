#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, copy, argparse
from controllable_channel_generation import get_pc_channel_sampler

sys.path.append('./')
from loaders import Channels
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
from sde_score.losses import get_optimizer
from sde_score.models.ema import ExponentialMovingAverage
from sde_score.utils import restore_checkpoint

sns.set(font_scale=2)
sns.set(style="whitegrid")
import sde_score.models
from sde_score.models import utils as mutils
from sde_score.models import ncsnv2, ncsnpp, layers, normalization
from sde_score.models import ddpm as ddpm_model
from sde_score.sde_lib import VESDE, VPSDE, subVPSDE
from channel_sampling import (ReverseDiffusionPredictor, LangevinCorrector)

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--train', type=str, default='mixed', help="Training scenario. Supported values: 'O1_28B', 'O1_28', 'I2_28B', 'mixed'")
    parser.add_argument('--test', type=str, default='mixed')
    parser.add_argument('--model_pth', type=str, default='checkpoint_20.pth', help="File name of the saved model")
    parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
    parser.add_argument('--pilot_alpha', nargs='+', type=float, default=0.6)
    args = parser.parse_args()

    # Disable TF32 due to potential precision issues
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    # GPU
    torch.cuda.set_device(args.gpu_id)

    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Number of validation channels
    """ In our paper, we set num_test_sample=256 to get smooth plots, but this will take longer inference time"""
    num_test_sample = 64

    # Target file
    target_dir = f'models/DM/{args.train}/checkpoints/'

    sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
    if sde.lower() == 'vesde':
        if args.train == 'O1_28':
            from configs.ve import CE_ncsnpp_deep_continuous as configs
        else:
            from configs.ve import CE_ncsnpp_deep_continuous_norm as configs
        ckpt_filename = os.path.join(target_dir, args.model_pth)
        if os.path.exists(ckpt_filename):
            print(f"Find the trained model at {ckpt_filename}")
        else:
            assert("The model cannot be found")
        config = configs.get_config()
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif sde.lower() == 'vpsde':
        raise NotImplementedError

    batch_size = num_test_sample  # @param {"type":"integer"}
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    # Load the trained diffusion model
    score_model = mutils.create_model(config)
    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    # Train and validation seeds
    train_seed, val_seed = 1111, 2222
    # Training dataset
    config.data.channel = args.train
    config.data.scenario_list = ['O1_28B', 'O1_28', 'I2_28B'] if args.train == 'mixed' else [args.train]
    dataset = Channels(train_seed, config, norm=config.data.norm_channels)
    # Validation dataset
    val_config = copy.deepcopy(config)
    val_config.data.channel = args.test
    val_config.data.scenario_list = ['O1_28B', 'O1_28', 'I2_28B'] if args.test=='mixed' else [args.test]
    val_config.data.num_pilots = int(np.floor(config.data.image_size[0] * args.pilot_alpha))
    val_config.data.spacing_list = args.spacing
    if args.train == 'O1_28':
        val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    else:
        val_dataset = Channels(val_seed, val_config, norm=config.data.norm_channels)


    # Range of SNR
    snr_range = np.arange(-10, 32.5, 2.5)
    noise_range = 10 ** (-snr_range / 10.) * config.data.image_size[0]

    # Construct file to save results
    nmse_all = np.zeros((len(snr_range), int(config.model.num_scales), num_test_sample))
    result_dir = f'./results/DM/test_train-{args.train}_test-{args.test}'
    os.makedirs(result_dir, exist_ok=True)

    # Pilot signals and channels
    val_loader = DataLoader(val_dataset, batch_size=num_test_sample, shuffle=True, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
    print('The validation datasets include %d validation channels' % len(val_dataset))
    sample = next(val_iter)
    Pilot_signal = sample['P'].cuda()  # pilot signal
    Pilot_herm = torch.conj(torch.transpose(Pilot_signal, -1, -2))  # Hermitian pilot signal
    val_H_herm_coefficients = sample[
        'H_herm'].cuda()  # (num_test_sample, 2, n_tx, n_rx)  # Hermitian channel coefficients
    val_H_herm = val_H_herm_coefficients[:, 0] + 1j * val_H_herm_coefficients[:, 1]  # Complex Hermitian channel matrix
    ground_truth = val_H_herm  # Ground truth channels
    shape = (batch_size, 2, 64, 16)


    # For different SNR scenarios
    for snr_idx, channel_noise_variance in enumerate(noise_range):
        # print(f"start estimating snr_idx {snr_idx} snr: {snr_range[snr_idx]}")
        # The received pilot signals
        Y_received = torch.matmul(Pilot_herm, val_H_herm)
        Y_received = Y_received + np.sqrt(channel_noise_variance) * torch.randn_like(Y_received)
        predictor = ReverseDiffusionPredictor  # @param ["NonePredictor", "EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor"] {"type": "raw"}
        corrector = LangevinCorrector  # @param ["NoneCorrector", "LangevinCorrector", "AnnealedLangevinDynamics"] {"type": "raw"}
        snr = 0.16
        n_steps = 2
        probability_flow = False
        pc_channel_sampler = get_pc_channel_sampler(sde, shape, predictor, corrector, snr, n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous)
        x, n, nmse = pc_channel_sampler(score_model, Y_received, Pilot_signal, channel_noise_variance, ground_truth)
        nmse_all[snr_idx, :] = nmse
        avg_nmse = np.mean(nmse_all, axis=-1)
        best_nmse = np.min(avg_nmse, axis=-1)

        print(f"snr [dB]: {snr_range[snr_idx]}, nmse [dB]: {10*np.log10(best_nmse[snr_idx])}")

        # Save results to file based on noise
        save_dict = {'nmse_all': nmse_all,
                     'avg_nmse': avg_nmse,
                     'best_nmse': best_nmse,
                     'spacing': args.spacing,
                     'pilot_alpha': args.pilot_alpha,
                     'snr_range': snr_range,
                     'val_config': val_config,
                     }
        torch.save(save_dict, os.path.join(result_dir, 'results.pt'))


    # Plot result
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 10))
    plt.plot(snr_range, 10 * np.log10(best_nmse), linewidth=4, label=f"{args.test}, DM (trained in {args.train})")
    plt.grid()
    plt.legend()
    plt.title('Channel estimation')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Normalized MSE [dB]')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'results.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

