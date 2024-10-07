import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    config.training.num_workers = 4
    training.n_iters = 1300001
    training.snapshot_freq = 10000 # 50
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.eval_pilot_snr = 0

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 10
    evaluate.end_ckpt = 100
    evaluate.batch_size = 32
    evaluate.enable_sampling = False
    evaluate.num_samples = 640
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'DeepMIMO'
    data.image_size = [64, 16]  # [Nt, Nr] for the transposed channel
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 2  # [Re, Im]
    data.num_pilots = data.image_size[1]
    data.num_paths = 10
    data.noise_std = 0
    data.norm_channels = 'global'
    data.spacing_list = [0.5]  # Training and validation

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 2100  # default: 1000
    model.beta_min = 0.1
    model.beta_max = 40.  # default: 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.00  # No weight decay
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8  # default [SDE]: 1e-8;  default [MIMO-CE]: 0.001
    optim.warmup = 5000
    optim.grad_clip = 1.
    config.optim.amsgrad = False

    config.seed = 42
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return config
