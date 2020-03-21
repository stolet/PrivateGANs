# PrivateGANs
Work on creating and evaluating generative adversarial networks with differentially private guarantees.

The GANs architecture used is from the paper "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
from ICLR 2018.

## Setup

```
# 1. Make sure the NVIDIA drivers for your GPU are installed

# 2. Install docker on your machine

# 3. Install NVIDIA docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 4. Check if GPU is available
lspci | grep -i nvidia

# 5. Verify Nvidia docker installation
docker run --gpus all --rm nvidia/cuda nvidia-smi

# 6. Download Tensorflow docker image
docker pull tensorflow/tensorflow:latest-gpu 

# 7. Build docker image (run inside the the PrivateGANs directory)
docker image build -t privategans:2.0 .

# 8. Run the container
docker container run -ti --gpus all privategans:2.0
