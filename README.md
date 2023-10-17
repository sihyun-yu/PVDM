## PVDM

Official PyTorch implementation of **["Video Probabilistic Diffusion Models in Projected Latent Space"](https://arxiv.org/abs/2302.07685)** (CVPR 2023).   
[Sihyun Yu](https://sihyun.me/)<sup>1</sup>, 
[Kihyuk Sohn](https://sites.google.com/site/kihyuksml/)<sup>2</sup>, 
[Subin Kim](https://subin-kim-cv.github.io/)<sup>1</sup>, 
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1</sup>.  
<sup>1</sup>KAIST, <sup>2</sup>Google Research  
[paper](https://arxiv.org/abs/2302.07685) | [project page](https://sihyun.me/PVDM/)

<p align="center">
    <img src=assets/ucf101_long.gif> 
    <img src=assets/sky_long.gif> 
</p>

---

### FAQ
**Q. Why is adversarial loss required?**  
**A.** This is for perceptual compression, just like in latent diffusion in the image domain.

**Q. What is the meaning of FVD in Tensorboard during training of diffusion models?**  
**A.** It is an FVD score between 16 real videos vs. generated videos, which is very noisy and **NOT the values** for reported evaluation (which uses 2,048 videos). Thus, it is recommended to track the status of training by visualizing the quality of videos. For UCF-101, we trained the model for 850k iterations with a batch size of 64.

**Q. Can PVDM be trained for text-to-video generation setup or on a more complex dataset?**  
**A.** Due to a lack of resources, we haven't tried it. However, there are several works that use PVDM and achieve reasonable results.
- HiP (text-to-video generation on robotics dataset) [[link](https://hierarchical-planning-foundation-model.github.io/)]
- Seer (text-conditioned video prediction) [[link](https://openreview.net/forum?id=qHGgNyQk31)]

**Q. Should I use an ema model for inference?**  
**A.** Yes, in most cases, the ema model shows much better performance.

---



### 1. Environment setup
```bash
conda create -n pvdm python=3.8 -y
conda activate pvdm
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy
```

### 2. Dataset 

#### Dataset download
Currently, we provide experiments for the following two datasets: [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [SkyTimelapse](https://github.com/weixiong-ur/mdgan). Each dataset should be placed in `/data` with the following structures below; you may change the data location directory in `tools/dataloadet.py` by adjusting the variable `data_location`.

#### UCF-101
```
UCF-101
|-- class1
    |-- video1.avi
    |-- video2.avi
    |-- ...
|-- class2
    |-- video1.avi
    |-- video2.avi
    |-- ...
    |-- ...
```

#### SkyTimelapse
```
SkyTimelapse
|-- train
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- video2
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
|-- val
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
```

### 3. Training

#### Autoencoder

First, execute the following script:
```bash
 python main.py 
 --exp first_stage \
 --id [EXP_NAME] \
 --pretrain_config configs/autoencoder/base.yaml \
 --data [DATASET_NAME] \
 --batch_size [BATCH_SIZE]
```
Then the script will automatically create the folder in `./results` to save logs and checkpoints.

If the loss converges, then execute the following script:
```bash
 python main.py 
 --exp first_stage \
 --id [EXP_NAME]_gan \
 --pretrain_config configs/autoencoder/base_gan.yaml \
 --data [DATASET] \
 --batch_size [BATCH_SIZE] \
 --first_stage_folder [DIRECTOTY OF PREVIOUS EXP]
```

Here, `[EXP_NAME]` is an experiment name you want to specifiy (string), `[DATASET]` is either `UCF101` or `SKY`, and `[DIRECTOTY OF PREVIOUS EXP]` is a directory for the previous script. For instance, the entire scripts for training the model on UCF-101 becomes: 
```bash
 python main.py \
 --exp first_stage \
 --id main \
 --pretrain_config configs/autoencoder/base.yaml \
 --data UCF101 \
 --batch_size 8

 python main.py \
 --exp first_stage \ 
 --id main_gan \
 --pretrain_config configs/autoencoder/base_gan.yaml \
 --data UCF101 \
 --batch_size 8 \
 --first_stage_folder 'results/first_stage_main_UCF101_42/'
```

You may change the model configs via modifying `configs/autoencoder`. Moreover, one needs early-stopping to further train the model with the GAN loss (typically 8k-14k iterations with a batch size of 8).

#### Diffusion model

```bash
 python main.py \
 --exp ddpm \
 --id [EXP_NAME] \
 --pretrain_config configs/autoencoder/base.yaml \
 --data [DATASET] \
 --first_model [AUTOENCODER DIRECTORY] 
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size [BATCH_SIZE]
```

Here, `[EXP_NAME]` is an experiment name you want to specifiy (string), `[DATASET]` is either `UCF101` or `SKY`, and `[DIRECTOTY OF PREVIOUS EXP]` is a directory of the autoencoder to be used. For instance, the entire scripts for training the model on UCF-101 becomes: 
```bash
 python main.py \
 --exp ddpm \
 --id main \
 --pretrain_config configs/autoencoder/base.yaml  \
 --data UCF101 \
 --first_model 'results/first_stage_main_gan_UCF101_42/model_last.pth'  
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size 64
```

### 4. Evaluation
Due to the code refactoring, we re-trained the models upon this codebase. We provide the corresponding checkpoints in the below:
|              | Autoencoder | Diffusion Model (400/20-s, eta=1) |
|--------------|-------------|-----------------------------------|
| UCF-101      | [link](https://drive.google.com/file/d/1L8l8h_SKN1BPm1P9cUhUmTsVZN0SCmqe/view?usp=drive_link)  | [link](https://drive.google.com/file/d/11EDh97-lq07xgvAx3P5KURQfrPtrcX0G/view?usp=drive_link)|
| SkyTimelapse | [link](https://drive.google.com/file/d/1fC_mUTZnxktR8GKfUnpB6L41sI5BsDxc/view?usp=drive_link)  | [link](https://drive.google.com/file/d/10ffoqZ9Zu0jVKZHFmrQQCiX9dM7FISj9/view?usp=drive_link)|


### Citation
```bibtex
@inproceedings{yu2023video,
  title={Video Probabilistic Diffusion Models in Projected Latent Space},
  author={Yu, Sihyun and Sohn, Kihyuk and Kim, Subin and Shin, Jinwoo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

### Reference
This code is mainly built upon [SiMT](https://github.com/jihoontack/simt), [latent-diffusion](https://github.com/CompVis/latent-diffusionn), and [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repositories.\
We also used the code from following repositories: [StyleGAN-V](https://github.com/universome/stylegan-v), [VideoGPT](https://github.com/wilson1yan/VideoGPT), and [MDGAN](https://github.com/weixiong-ur/mdgan).


