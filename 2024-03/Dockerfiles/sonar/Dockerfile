# base image
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

LABEL maintainer "Natanael Moura Junior <natmourajr@lps.ufrj.br>"

USER root
ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen


RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    vim \ 
    less \
    nano \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set python 3 as the default python
# RUN update-alternatives --set python /usr/local/bin/python3
RUN apt-get update -y
RUN apt-get install -y libsndfile-dev

# Install pip packages
RUN pip install numpy==1.26.4 scipy==1.14.0 pandas==2.2.2 scikit-learn==1.5.1 soundfile==0.12.1 librosa==0.10.2.post1