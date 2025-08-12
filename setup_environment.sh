#!bin/bash

# create enviroment using Miniconda (or Anaconda)
conda create -n continual_clip python=3.9
conda activate continual_clip

# install pytorch
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install peft
pip install -r requirements.txt

# install CLIP
pip install git+https://github.com/openai/CLIP.git

