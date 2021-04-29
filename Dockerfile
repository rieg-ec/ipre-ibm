FROM nvidia/cuda:11.1-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy \
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
    pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip3 install torch-geometric


RUN ["mkdir", "notebooks"]
COPY conf/.jupyter /root/.jupyter

COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006


CMD ["/run_jupyter.sh"]

