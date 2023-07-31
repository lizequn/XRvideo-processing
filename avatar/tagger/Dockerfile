FROM onnx

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
#RUN rm /etc/apt/sources.list.d/cuda.list \
#    && rm /etc/apt/sources.list.d/nvidia-ml.list \
#    && apt-key del 7fa2af80 \
#    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev python3-opencv

WORKDIR /work

COPY ./requirements.txt /work

RUN pip install -r requirements.txt

RUN conda install cudatoolkit

RUN pip install onnxruntime-gpu
