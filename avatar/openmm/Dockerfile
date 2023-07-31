FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /work

RUN git clone https://github.com/open-mmlab/playground

RUN cd playground/label_anything && \
    pip install opencv-python pycocotools matplotlib onnxruntime onnx && \
    pip install git+https://github.com/facebookresearch/segment-anything.git

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

RUN pip install label-studio==1.7.3 &&\ 
    pip install label-studio-ml==1.0.9
ENV ML_TIMEOUT_SETUP=40
ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
ENV LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/work/data

RUN apt-get update &&\
    apt-get install -y supervisor &&\
    mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8003 8080
CMD ["/usr/bin/supervisord"]