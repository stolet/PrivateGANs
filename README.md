# PrivateGANs
Work on creating and evaluating generative adversarial networks with differentially private guarantees.

The GANs architecture used is from the paper "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
from ICLR 2018.

## Setup

### With container

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
docker pull tensorflow/tensorflow:2.1.0

# 7. Build docker image (run inside the the PrivateGANs directory)
docker image build -t privategans:2.0 .

# 8. Run the container
docker container run -ti --gpus all privategans:2.0

```

### Without container

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-430
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.4.38-1+cuda10.1  \
    libcudnn7-dev=7.6.4.38-1+cuda10.1


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

```
