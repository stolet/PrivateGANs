# Set up the image for tensorflow 2.1 with gpu support
FROM tensorflow/tensorflow:2.1.0

# Set the working directory for the docker image
WORKDIR /usr/src/app

# Copy the Private GAN directory
COPY . .

# Install requirements
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements-pip.txt
