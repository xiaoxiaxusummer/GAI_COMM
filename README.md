# GAI_COMM

This is Pytorch code Implementation of paper "Generative Artificial Intelligence (GAI) for Mobile Communications: A Diffusion Model Perspective", which has been accepted by IEEE Communications Magzine (COMMAG).

## Dataset Preparation
We consider [DeepMIMO dataset](https://www.deepmimo.net/versions/5gnr/), including 
  + 2 outdoor scenes "O1_28" and "O1_28B"
  + 1 indoor scene "I2-28B"
  + their mixed scenes "mixed" <br>

**You may directly download [our datasets here](https://drive.google.com/drive/folders/1AwuStJrzWd1K4oBgbI8u3ua5IMKYO-PZ?usp=sharing). Then, put the obtained .mat channel data in folder [`DeepMIMO-5GNR/DeepMIMO_dataset`](./DeepMIMO-5GNR/DeepMIMO_dataset).**
> If you'd like to prepare your own datasets, please download [ray tracing data](https://www.deepmimo.net/scenarios/) provided by DeepMIMO and follow their instructions. We also provide [an example script](./DeepMIMO-5GNR/DeepMIMO_Dataset_Generator.m) to generate datasets.


## Conditional Diffusion Model Training and Test
To train your own diffusion model based on SDE, please run [`train_diffusion_model.py`](./train_diffusion_model.py) after prepare the dataset. 
The trained diffusion model can be tested by runnig [`test_diffusion_model.py`](./test_diffusion_model.py). <br>

## Our Pretrained Models 
**We provide `.pth` file of our conditional diffusion models trained in ["mixed"](https://drive.google.com/drive/folders/1Gu5Vyj8VIYAKS48T3Ol3WUI-YhiXfRza?usp=sharing) and ["O1_28"](https://drive.google.com/drive/folders/1_BZ831vk7W25xhw49ta6h5hI-si12cwx?usp=sharing) scenes, respectively. <br>**
**After downloading the ``.pth`` file, please put them in folder** [`./models/DM/`](./models/DM/) and then run [`test_diffusion_model.py`](./test_diffusion_model.py).
### Configurations
+ In [`test_diffusion_model.py`](./test_diffusion_model.py), we set `num_test_sample=64` (see line 54) to accelerate the conditional generating process. <br>
  > You may use `num_test_sample=256` to obtain smoother plots and reproduce results in our paper, but this will take longer inference time.
  Hyperparameters can be customized by modifying ['./configs/ve/CE_ncsnpp_deep_continuous.py'](./configs/ve/CE_ncsnpp_deep_continuous.py). Note that this may significantly impact the generative performance.
### Algorithm Design and Future Works
+ The conditioned channel generation by predictor-corrector sampling is implemented in [`controllable_channel_generation.py`](./controllable_channel_generation.py).
+ The current project utilizes score-based SDE, which requires 2000+ sampling steps to obtain desirable channel estimation. In order to reduce the sampling time, few-step diffusion techniques (which only requires 1-4 denoising steps) and latent diffusion models could be further investigated in future works.



