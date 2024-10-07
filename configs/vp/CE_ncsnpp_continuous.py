# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VP SDE."""
from configs.default_CE_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  config.training = training

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  config.sampling = sampling

  # data
  data = config.data
  # data.centered = True
  config.data = data
  config.data.channels = 2  # {Re, Im}
  config.data.noise_std = 0
  config.data.image_size = [16, 64]  # [Nr, Nt] for the transposed channel
  config.data.num_pilots = config.data.image_size[1]
  config.data.norm_channels = 'global'
  config.data.spacing_list = [0.5]  # Training and validation
  config.data.num_paths = 10


  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  config.model = model
  config.model.ema = True
  config.model.ema_rate = 0.999
  config.model.normalization = 'InstanceNorm++'
  config.model.nonlinearity = 'elu'
  config.model.sigma_dist = 'geometric'
  config.model.num_classes = 2311  # Number of train sigmas and 'N'
  config.model.ngf = 32


  # Training
  config.training.batch_size = 32
  config.training.num_workers = 4
  config.training.n_epochs = 400
  config.training.anneal_power = 2
  config.training.log_all_sigmas = False

  return config
