"""
Training diffusion model by score-based SDE
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"
import sde_score.run_lib as run_lib
import logging
import tensorflow as tf
import torch, os, argparse

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

def main(args):
  # Create the working directory
  tf.io.gfile.makedirs(args.workdir)
  # Set logger so that it outputs to both console and file
  # Make logging work for both disk and Google Cloud Storage
  gfile_stream = open(os.path.join(args.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  # Run the training pipeline
  run_lib.train(args.config, args.workdir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', type=int, default=1)  # Assign the gpu device idx
  parser.add_argument('--train', type=str, default='mixed', help="Training scenario. Supported values: 'O1_28B', 'O1_28', 'I2_28B', 'mixed'")
  parser.add_argument('--scenario_list', type=list, default=['O1_28B', 'O1_28', 'I2_28B']) # default=['O1_28B', 'O1_28', 'I2_28B'])
  parser.add_argument('--workdir', type=str, default='models/DM/I2_28B/')
  parser.add_argument('--eval_folder', type=str, default='eval')
  args = parser.parse_args()
  args.scenario_list = ['O1_28B', 'O1_28', 'I2_28B'] if args.train=='mixed' else [args.train]
  if args.train == 'O1_28':
    from configs.ve.CE_ncsnpp_deep_continuous import get_config
  else:
    from configs.ve.CE_ncsnpp_deep_continuous_norm import get_config

  config = get_config()

  # Choose GPU
  torch.cuda.set_device(args.gpu_id)
  config.device = torch.device('cuda', args.gpu_id)

  # Choose channel scenario
  config.data.train_scenario = args.train
  config.data.scenario_list = args.scenario_list
  args.config = config

  main(args)

