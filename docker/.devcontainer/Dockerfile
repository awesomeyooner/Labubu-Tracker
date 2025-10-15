# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04
# FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    curl \
    git \
    nano \
    software-properties-common \
    libceres-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils \
    libx11-dev \
    libxrender-dev \
    libxtst-dev \
    pciutils \
    python3 \
    python3-venv \
    python3-pip 
    
CMD ["/bin/bash"]

ARG USERNAME="user"
ARG PASSWORD="password"

# Create new user
RUN useradd -ms /bin/bash ${USERNAME} \
    && echo "${USERNAME}:${PASSWORD}" | chpasswd \
    && usermod -aG sudo ${USERNAME} \
    && usermod -aG video ${USERNAME} \
    && adduser ${USERNAME} sudo \
    && adduser ${USERNAME} video \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to new user
USER ${USERNAME}

# Switch main workspace to $HOME of user
WORKDIR /home/${USERNAME}

# Start doing things below!

# Scripts
RUN mkdir scripts

# Install Anaconda
COPY scripts/anaconda.sh scripts/anaconda.sh
RUN sudo chmod +x scripts/anaconda.sh && ./scripts/anaconda.sh

RUN rm -rf scripts