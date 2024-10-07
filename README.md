# Generative Artificial Intelligence (GAI) for Mobile Communications: A Diffusion Model Perspective

This is Pytorch code Implementation of paper **"Generative Artificial Intelligence (GAI) for Mobile Communications: A Diffusion Model Perspective"**, which has been accepted by IEEE Communications Magzine (COMMAG).


## Recommended Environment
>+ torch==2.4.0+cu121, torchaudio==2.4.0+cu121, torchvision==0.19.0+cu121, xformers==0.0.27.post2
>+ tensorflow-datasets==4.9.2, tensorflow-gpu==2.9.0, tensorflow-gan==2.1.0, tensorflow-probability==0.11.0, tensorboard==2.9.0
>+ jax==0.4.11, jaxlib==0.4.11, jinja2==3.1.3, ninja==1.11.1.1, torch2jax==0.0.1
>+ **[Notes]** To custom Pytorch CUDA Kernel, run [test_upfirdn2d.py](./sde_score/op/test/test_upfirdn2d.py) after installing the required packages. 

## Dataset Preparation
> We consider the following ray-tracing scenes available in [DeepMIMO dataset](https://www.deepmimo.net/versions/5gnr/) 
  > + 2 outdoor scenes: "O1_28" and "O1_28B"
  > + 1 indoor scene: "I2-28B"
  > + the mixed scenes: "mixed" <br>

**You may directly download [our datasets here](https://drive.google.com/drive/folders/1AwuStJrzWd1K4oBgbI8u3ua5IMKYO-PZ?usp=sharing). Then, put the obtained .mat channel data in folder [`DeepMIMO-5GNR/DeepMIMO_dataset`](./DeepMIMO-5GNR/DeepMIMO_dataset).**
> If you'd like to prepare your own datasets, download [DeepMIMO ray-tracing data](https://www.deepmimo.net/scenarios/) and follow their instructions. <br>
> We also provide the [script](./DeepMIMO-5GNR/DeepMIMO_Dataset_Generator.m) to generate and save our datasets.


## Training and Test
Train the score SDE based diffusion model through [`train_diffusion_model.py`](./train_diffusion_model.py) after preparing the dataset. <br>
Test the conditional diffusion model by runnig [`test_diffusion_model.py`](./test_diffusion_model.py). <br>

## Our Pretrained Checkpoints 
**Checkpoints of our conditional diffusion models trained in "mixed" and "O1_28" scenes are available at [`mixed/checkpoints/`](https://drive.google.com/drive/folders/1Gu5Vyj8VIYAKS48T3Ol3WUI-YhiXfRza?usp=sharing) and [`O1_28/checkpoints/`](https://drive.google.com/drive/folders/1_BZ831vk7W25xhw49ta6h5hI-si12cwx?usp=sharing).
<br>**
>**[Notes]** After downloading the ``.pth`` file, put them in folder [`./models/DM/mixed/checkpoints`](./models/DM/mixed/checkpoints) (or [`./models/DM/O1_28/checkpoints`](./models/DM/O1_28/checkpoints)) and then run [`test_diffusion_model.py`](./test_diffusion_model.py).
### Configurations
```
  test_diffusion_model.py
    --gpu_id: index of gpu device
    --train: scene for model training ("mixed", "O1_2B", or "I2_28B")
    --test: scene for model test
    --model_pth: File name of the saved model (e.g., 'XXXX.pth')
```
+ In [`test_diffusion_model.py`](./test_diffusion_model.py), we set `num_test_sample=64` (see line 53) to accelerate the conditional generating process. <br>
  > You may use `num_test_sample=256` to obtain smoother plots and reproduce results in our paper, but this will take longer inference time.
+ Hyperparameters can be customized by modifying ['./configs/ve/CE_ncsnpp_deep_continuous.py'](./configs/ve/CE_ncsnpp_deep_continuous.py). Note that this may significantly impact the generative performance.
### Algorithm Design and Future Works
+ The conditioned channel generation by predictor-corrector sampling is implemented in [`controllable_channel_generation.py`](./controllable_channel_generation.py).
+ The current project utilizes score-based SDE, which requires 2000+ sampling steps to obtain desirable channel estimation. In order to reduce the sampling time, few-step diffusion techniques (which only requires 1-4 denoising steps) and latent diffusion models could be further investigated in future works.

## Reference
If you find the code useful for your research, please consider citing
> X. Xu, X. Mu, Y. Liu, H. Xing, Y. Liu, A. Nallanathan, ``Generative Artificial Intelligence (GAI) for Mobile Communications: A Diffusion Model Perspective'', IEEE Communications Magazine, accepted, Sept. 2024.

