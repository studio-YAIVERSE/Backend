FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    xvfb

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install Django==4.1.7 \
    django-cors-headers==3.14.0 \
    django-extensions==3.2.1 \
    djangorestframework==3.14.0 \
    drf-yasg==1.21.5 \
    pillow==9.4.0 \
    gunicorn==20.1.0 \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    ninja imageio imageio-ffmpeg==0.4.4 xatlas gdown \
    git+https://github.com/NVlabs/nvdiffrast/ \
    meshzoo ipdb gputil h5py point-cloud-utils pyspng==0.1.0 \
    urllib3 scipy click tqdm opencv-python==4.5.4.58 \
    trimesh==3.21.5 "pyglet<2" xvfbwrapper==0.2.9 \
    ftfy git+https://github.com/openai/CLIP.git \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN imageio_download_bin freeimage
