#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The controllable channel generation is achieved by classifier-guidance based conditional DM.
Predictor-corrector sampling has been implemented here, which modifies the original jax implementation of conditional SDE by Song.
"""
from sde_score.models import utils as mutils
import torch
from channel_sampling import NoneCorrector, NonePredictor
import functools
from tqdm import tqdm as tqdm

def map_complex_to_coeff(X):
    """
    :param X: (batch_size, n_tx, n_rx)
    :return: X_coeff: (batch_size, 2, n_tx, n_rx)
    """
    return torch.view_as_real(X).permute(0,3,1,2)

def map_coeff_to_complex(X_coeff):
    """
    :param: X_coeff: (batch_size, 2, n_tx, n_rx)
    :return X: (batch_size, n_tx, n_rx)
    """
    return torch.view_as_complex(X_coeff.permute(0,2,3,1).contiguous())

def condition_grad_fn(X_coefficients, Y, Pilot, channel_noise):
    """Return gradient of log_{H} P(Y|H)"""
    X = map_coeff_to_complex(X_coefficients) # (batch_size, n_tx, n_rx)
    Pilot_herm = torch.conj(torch.transpose(Pilot, -1, -2))
    log_grad = torch.matmul(Pilot, Y-torch.matmul(Pilot_herm, X))
    conditional_grad = log_grad/channel_noise # (batch_size, n_tx, n_rx)
    # conditional_grad = map_complex_to_components(conditional_grad) # (batch_size, 2, n_tx, n_rx)
    conditional_grad = map_complex_to_coeff(conditional_grad) # (batch_size, 2, n_tx, n_rx)
    return conditional_grad

def conditional_predictor_update_fn(X_coefficients, t, Y, Pilot, channel_noise, model, sde, predictor, continuous, probability_flow):
  """
  The predictor update function for class-conditional sampling.
  X_coefficients:  real/image parts of H, [batch_size, 2, n_tx, n_rx]
  Y:  received measurement of pilot signals
  P:  pilot signal
  """
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  def total_grad_fn(X_coefficients, t):
    """:param X_coefficients: # (batch_size, 2, n_tx, n_rx) """
    score = score_fn(X_coefficients, t)  # Score of real/img parts of H_herm, [batch_size, 2, n_tx, n_rx]
    cond_score = condition_grad_fn(X_coefficients, Y, Pilot, channel_noise) # Score of likelihood P(Y|X), [batch_size, n_tx, n_rx]
    omega = 1/cond_score.abs().mean()*score.abs().mean()*5
    return score + cond_score*omega # scale to the same order of magnitude

  if predictor is None:
    predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, total_grad_fn, probability_flow)

  return predictor_obj.update_fn(X_coefficients, t)

def conditional_corrector_update_fn(X_coefficients, t, Y, Pilot, channel_noise, model, sde, corrector, continuous, snr, n_steps):
  """The corrector update function for class-conditional sampling."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

  def total_grad_fn(X_coefficients, t):
    """:param X_coefficients: # (batch_size, 2, n_tx, n_rx) """
    score = score_fn(X_coefficients, t)  # Score of real/img parts of H, [batch_size, 2, n_tx, n_rx]
    cond_score = condition_grad_fn(X_coefficients, Y, Pilot, channel_noise)  # Score of likelihood P(Y|X), [batch_size, 2, n_tx, n_rx]
    return score + cond_score

  if corrector is None:
    corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
  return corrector_obj.update_fn(X_coefficients, t)

def get_pc_conditional_sampler(sde, shape, predictor, corrector, snr,
                               n_steps=1, probability_flow=False,
                               continuous=False, denoise=True, eps=1e-5, device='cuda'):
  """Class-conditional sampling with Predictor-Corrector (PC) samplers.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    classifier: A `torch.Module` object that represents the architecture of the noise-dependent classifier.
    classifier_params: A dictionary that contains the weights of the classifier.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.
  Returns: A class-conditional image sampler.
  """
  # # A function that gives the logits of the noise-dependent classifier
  # logit_fn = mutils.get_logit_fn(classifier, classifier_params)
  # # The gradient function of the noise-dependent classifier
  # condition_grad_fn = mutils.get_condition_grad_fn(logit_fn)

  predictor_update_fn = functools.partial(conditional_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(conditional_corrector_update_fn,
                                          sde=sde,
                                          corrector = corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)


  def pc_conditional_sampler(model, Y, Pilot, channel_noise, ground_truth):
    """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

    Args:
      model: A score model.
      Y: Measurements of each sample.
    Returns:
      Class-conditional samples.
    """
    with torch.no_grad():
      X_coefficients = sde.prior_sampling(shape).to(device) # (batch_size, 2, n_tx, n_rx)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      nmse_log = []
      for i in tqdm(range(sde.N)):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        X_coefficients, X_mean = corrector_update_fn(X_coefficients=X_coefficients, t=vec_t, Y=Y, Pilot=Pilot, channel_noise=channel_noise, model=model)
        X_coefficients, X_mean = predictor_update_fn(X_coefficients=X_coefficients, t=vec_t, Y=Y, Pilot=Pilot, channel_noise=channel_noise, model=model)

        X_test = X_mean if denoise else X_coefficients
        X_test = map_coeff_to_complex(X_test)
        val_nmse_log = (torch.sum(torch.square(torch.abs(X_test - ground_truth)), dim=(-1, -2)) / \
                        torch.sum(torch.square(torch.abs(ground_truth)),
                                  dim=(-1, -2))).cpu().numpy()
        nmse_log.append(val_nmse_log)


      return X_test, sde.N * (n_steps + 1), nmse_log
  return pc_conditional_sampler




