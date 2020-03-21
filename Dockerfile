# Set up the image for tensorflow 1.15 with gpu support
FROM tensorflow/tensorflow:1.15.0-gpu

# Set the working directory for the docker image
WORKDIR /usr/src/app

# Copy the Private GAN directory
COPY . .
