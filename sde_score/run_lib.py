
"""
Training diffusion model for wireless channel generation.
"""

import os, copy, sys
import numpy as np
import tensorflow as tf
import logging

# Keep the import below for registering all model definitions
from sde_score.models import ddpm, ncsnv2, ncsnpp
from sde_score import losses, sde_lib
from sde_score.models import utils as mutils
from sde_score.models.ema import ExponentialMovingAverage
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from sde_score.utils import save_checkpoint, restore_checkpoint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from loaders import Channels, map_complex_to_components, map_components_to_complex
import channel_sampling as sampling
from controllable_channel_generation import get_pc_conditional_sampler

def train(config, workdir):
  """Runs the training pipeline.

  @:param
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  os.makedirs(os.path.dirname(checkpoint_meta_dir),exist_ok=True)

  # Resume training when intermediate checkpoints are detected
  if config.training.resume:
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_seed, eval_seed = 1111, 2222 # Seeds for train and test datasets
  dataset = Channels(train_seed, config, norm=config.data.norm_channels)
  dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                          shuffle=True, num_workers=config.training.num_workers, drop_last=True)
  train_iter = iter(dataloader)

  # Validation data
  eval_config = copy.deepcopy(config)
  eval_config.data.spacing_list = [config.data.spacing_list[0]]
  eval_dataset = Channels(eval_seed, eval_config, norm=[dataset.mean, dataset.std])
  eval_loader  = DataLoader(eval_dataset, batch_size=config.eval.batch_size, shuffle=False, num_workers=0, drop_last=True)
  eval_iter = iter(eval_loader)  # For validation

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  print("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Compute loss by Hermitian channels
    try:
      sample = next(train_iter)['H_herm'] # (batch_size, 2, n_tx, n_rx)
    except:
      train_iter = iter(dataloader)
      sample = next(train_iter)['H_herm']  # (batch_size, 2, n_tx, n_rx)
    # Convert data to torch arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(sample.numpy()).to(config.device).float()

    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      writer.add_scalar("training_loss", loss, step)
      print(f"step: {step}, training_loss: {loss.item()}")

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_losses = []
      for idx in range(len(config.data.spacing_list)):
        try:
          eval_sample = next(eval_iter) # (batch_size, 2, n_tx, n_rx)
        except:
          eval_iter = iter(eval_loader)
          eval_sample = next(eval_iter) # (batch_size, 2, n_tx, n_rx)
        eval_batch = torch.from_numpy(eval_sample['H_herm'].numpy()).to(config.device).float()
        eval_loss = eval_step_fn(state, eval_batch)
        eval_losses.append(eval_loss.item())
      logging.info("step: %d, eval_loss: %.5e" % (step, np.array(eval_losses).mean()))
      writer.add_scalar("eval_loss", np.array(eval_losses).mean(), step)
      print(f"step: {step}, eval_loss: {np.array(eval_losses).mean()}")

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save conditional samples
      if config.training.snapshot_sampling:
        try:
          eval_sample = next(eval_iter)  # (batch_size, 2, n_tx, n_rx)
        except:
          eval_iter = iter(eval_loader)
          eval_sample = next(eval_iter) # (batch_size, 2, n_tx, n_rx)
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size[0], config.data.image_size[1])

        """ 
        NOTE: To test the channel generative performance, please run test_diffusion_model.py 
        We select NoneCorrector here for accelerating training process.
        """
        predictor = sampling.ReverseDiffusionPredictor  # @param ["NonePredictor", "EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor"]
        corrector = sampling.NoneCorrector  # @param ["NoneCorrector", "LangevinCorrector"]

        snr = 0.16  # @param {"type": "number"}
        n_steps = 1  # @param {"type": "integer"}
        probability_flow = False  # @param {"type": "boolean"}

        pc_conditional_sampler = get_pc_conditional_sampler(sde, sampling_shape, predictor, corrector, snr,
                                                                     n_steps=n_steps, probability_flow=probability_flow,
                                                                     continuous=config.training.continuous)

        # Get all validation pilots and channels
        Pilot = eval_sample['P'].to(config.device) # (batch_size, n_tx, n_rx)
        Pilot_herm = torch.conj(torch.transpose(Pilot, -1, -2)) # (batch_size, n_rx, n_tx)
        eval_H_herm_coefficients = eval_sample['H_herm'].to(config.device) # (batch_size, 2, n_tx, n_rx)
        eval_H_herm = map_components_to_complex(eval_H_herm_coefficients) # (batch_size, n_tx, n_rx)
        channel_noise = 10 ** (config.training.eval_pilot_snr/10.) * config.data.image_size[1]
        Y_received = torch.matmul(Pilot_herm, eval_H_herm)
        Y_received = Y_received + np.sqrt(channel_noise) * torch.randn_like(Y_received)
        ground_truth = eval_H_herm

        x, n, nmse = pc_conditional_sampler(score_model, Y_received, Pilot, channel_noise, ground_truth)
        ema.restore(score_model.parameters())

        logging.info(f"step: {step}, avg nmse: {np.array(nmse).mean(-1)}, best nmse: {np.array(nmse).mean(-1).min(-1)}.")
        print(f"step: {step}, avg nmse: {np.array(nmse).mean(-1)}, best nmse: {np.array(nmse).mean(-1).min(-1)}.")