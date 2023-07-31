FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=True
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0"
ENV CUDA_HOME=/usr/local/cuda-11.6/

RUN mkdir -p /home/zequn/Grounded-Segment-Anything

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano=2.* \
    vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /home/zequn/Grounded-Segment-Anything
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git .
RUN python -m pip install --no-cache-dir -e segment_anything && \
    python -m pip install --no-cache-dir -e GroundingDINO
WORKDIR /home/zequn
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai tqdm

RUN pip uninstall -y supervision
RUN pip install -q supervision==0.6.0
# download weights
ENV GROUNDING_DINO_CONFIG_PATH=/home/zequn/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
RUN mkdir -p /home/zequn/weights
WORKDIR /home/zequn/weights

RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

ENV GROUNDING_DINO_CHECKPOINT_PATH=/home/zequn/weights/groundingdino_swint_ogc.pth
ENV SAM_CHECKPOINT_PATH=/home/zequn/weights/sam_vit_h_4b8939.pth