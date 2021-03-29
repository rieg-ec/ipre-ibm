FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install tensorflow && \
    pip3 install numpy \
        pandas \
        sklearn \
        matplotlib \
        seaborn \
        jupyter \
        pyyaml \
        mayavi \
        scipy  \
        pyreadr \
        meshio \
        h5py && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils && \
    pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


RUN ["mkdir", "notebooks"]
COPY conf/.jupyter /root/.jupyter

COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006


CMD ["/run_jupyter.sh"]

