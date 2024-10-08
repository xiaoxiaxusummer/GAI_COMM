#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:32:31 2021

@author: marius
"""

import os, torch, sys, itertools, copy, argparse
from matplotlib import pyplot as plt
import numpy as np

sys.path.append('../')

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

num_threads = 2
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OMP_DYNAMIC"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
torch.set_num_threads(num_threads)

from tqdm import tqdm as tqdm
from loaders import Channels
from torch.utils.data import DataLoader

# Config args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='mixed')
parser.add_argument('--test', type=str, default='mixed')
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

if args.train == 'O1_28':
    from configs.ve import CE_ncsnpp_deep_continuous as configs
else:
    from configs.ve import CE_ncsnpp_deep_continuous_norm as configs

config = configs.get_config()


args.antennas = config.data.image_size
config.data.spacing_list = [args.spacing[0]]

####################################### Prepare datasets #######################################
train_seed, val_seed = 1111, 2222

# Number of samples
num_test_sample = 256

# Load training dataset
config.data.channel = args.train
if not args.train == 'mixed':
    config.data.scenario_list = [args.train]
else:
    config.data.scenario_list = ['O1_28B', 'O1_28', 'I2_28B']
dataset = Channels(train_seed, config, norm='global')

# Load validation dataset
val_config = copy.deepcopy(config)
val_config.purpose = 'val'
val_config.data.channel = args.test
if not args.test == 'mixed':
    val_config.data.scenario_list = [args.test]
else:
    val_config.data.scenario_list = ['O1_28B', 'O1_28', 'I2_28B']
val_config.data.spacing_list = [args.spacing]
val_config.data.num_pilots = int(np.floor(args.antennas[1] * args.pilot_alpha))
val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
val_loader = DataLoader(val_dataset, batch_size=num_test_sample, shuffle=True, num_workers=0, drop_last=True)
val_iter = iter(val_loader)

# Noise power
snr_range = np.asarray(np.arange(-10, 32.5, 2.5))
noise_range = 10 ** (-snr_range / 10.)

# Results logging
nmse_all = np.zeros((len(snr_range), num_test_sample))
result_dir = os.path.join(PROJECT_DIR, f'results/ml_baseline/train{args.train}_test{args.test}')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# Get a batch of sample
val_sample = next(val_iter)
del val_iter, val_loader
val_P = val_sample['P']
val_P = torch.conj(torch.transpose(val_P, -1, -2))
val_H_herm = val_sample['H_herm']
val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
# Convert to numpy vectors
val_P = val_P.resolve_conj().numpy()
val_H = val_H.resolve_conj().numpy()

# For each SNR value
for snr_idx, local_noise in tqdm(enumerate(noise_range)):
    val_Y = np.matmul(val_P, val_H)
    val_Y = val_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * \
            (np.random.normal(size=val_Y.shape) + \
             1j * np.random.normal(size=val_Y.shape))

    # For each sample
    for sample_idx in tqdm(range(val_Y.shape[0])):
        # Normal equation
        normal_P = np.matmul(val_P[sample_idx].T.conj(), val_P[sample_idx]) + \
                   local_noise * np.eye(val_P[sample_idx].shape[-1])
        normal_Y = np.matmul(val_P[sample_idx].T.conj(), val_Y[sample_idx])
        # Single-shot solve
        est_H, _, _, _ = np.linalg.lstsq(normal_P, normal_Y)
        # Estimate error
        nmse_all[snr_idx, sample_idx] = \
            (np.sum(np.square(np.abs(est_H - val_H[sample_idx])), axis=(-1, -2))) / \
            np.sum(np.square(np.abs(val_H[sample_idx])), axis=(-1, -2))

avg_nmse = np.mean(nmse_all, axis=-1)
# Plot results
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
plt.plot(snr_range, avg_nmse[0, 0], linewidth=4, label=f'{args.test}, Maximum likelihood')
plt.grid()
plt.legend()
plt.title('Channel estimation')
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results_mse.png'), dpi=300,
            bbox_inches='tight')
plt.close()

# Save to file
torch.save({'snr_range': snr_range,
            'spacing': args.spacing,
            'pilot_alpha': args.pilot_alpha,
            'nmse_all': nmse_all,
            'avg_nmse': avg_nmse
            }, result_dir + f'/results_Nt{args.data.image_size[0]}_Nr{args.data.image_size[1]}.pt')
