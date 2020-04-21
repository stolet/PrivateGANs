# PrivateGANs
Work on creating and evaluating generative adversarial networks with differentially private guarantees.

The GANs architecture used is from the paper "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
from ICLR 2018.

## Setup CUDA 10.0 and Libcudnn 7

Remove all Nvidia traces from machine 

```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
```

Setup correct CUDA ppa on system

```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

Install CUDA 10.0 Packages

```
sudo apt update
sudo apt install cuda-10-0
sudo apt install libcudnn7
```

Specify path to CUDA in profile

```
sudo vi ~/.profile
```

Add following lines to the end of the profile file:

```
if [ -d "/usr/local/cuda-10.0/bin/" ]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

Restart machine

## Setup Python environment

```
sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv environment
source environment/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install setuptools -U
sudo apt-get install libffi-dev python-dev build-essential
```

## Install Anaconda

Necessary to install lmdb

```
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

Reboot machine

## Install packages

```
python3 -m pip install pandas
python3 -m pip install tensorflow
python3 -m pip install tensorflow-gpu==1.15
conda install lmdb
cd Projects/PrivateGANs/progressive_growing_of_gans/
python3 -m pip install -r requirements-pip.txt
```


