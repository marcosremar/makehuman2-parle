#!/bin/bash
# Runs inside the PyTorch container: Python packages into node scratch, then the test.
set -uo pipefail
export PYTHONUSERBASE=$WORKDIR/py PATH=$WORKDIR/py/bin:$PATH
pip install -q --user "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" "huggingface_hub==0.26.5" trimesh pymeshlab rembg onnxruntime opencv-python-headless scikit-image pygltflib einops omegaconf
pip install -q --user --no-deps https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip
python3 -c "import torch; print('TORCH', torch.__version__, torch.cuda.get_device_name(0))"
python3 /home/maramosp/cabeca-teste/cabeca_teste.py /home/maramosp/cabeca-teste/saida
