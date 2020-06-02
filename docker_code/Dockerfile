FROM  nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER ionishi


RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python-tk \
    python-opencv \
    libopenblas-dev \
    python-scipy \
    python-nose \
    python-matplotlib \
    python-shapely \
    git \
    vim \
    libjpeg-dev libpng-dev \
    python-imaging \
    wget \
    gcc g++ \
    git-core 
    
RUN apt-get update
RUN apt-get install -y \
    file \
    dos2unix \
    unzip \
    graphviz python-tk protobuf-compiler python-pil python-lxml

# install anaconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN pip install h5py imageio pillow tqdm \
	pydot pydot-ng graphviz opencv-python imageio seaborn cvxopt 
	
RUN conda install -c menpo opencv3=3.1.0 -y

WORKDIR /home

ENV HOME=/home/$USER \
    SHELL=/bin/bash

CMD bash



