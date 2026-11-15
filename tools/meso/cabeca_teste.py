"""Quick test: a stylized head image (SDXL-Turbo, text only) and its 3D shape (Hunyuan3D-2mini, shape only). Writes PNGs and GLBs to OUT."""
import os
import sys

import torch
from diffusers import AutoPipelineForText2Image

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
PROMPT = ("3D render of a stylized cartoon character head from a French animated TV series for children, young adult man, "
          "long oval face, big round bulbous nose, thick dark eyebrows, round friendly eyes, wide simple mouth closed, bald head, "
          "smooth matte clay skin, neutral expression, front view, centered, looking at camera, plain white background, soft studio light")
NEG = "hair, hat, glasses, open mouth, teeth, side view, text, watermark, realistic photo, wrinkles"
t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
images = []
import glob
DONE = sorted(glob.glob(f"{OUT}/homem-*.png"))
for seed in ([] if DONE else range(6)):
    image = t2i(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=4, guidance_scale=0.0, height=512, width=512,
                generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    path = f"{OUT}/homem-{seed}.png"
    image.save(path)
    images.append(path)
    print("IMAGEM", path, flush=True)
images = images or DONE
del t2i
torch.cuda.empty_cache()

import types
# pymeshlab needs libX11, absent in the container; only its face reducer uses it, and the test does not.
stub = types.ModuleType('pymeshlab')
stub.MeshSet = stub.Mesh = object
sys.modules['pymeshlab'] = stub
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

remover = BackgroundRemover()
shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", use_safetensors=True, device="cuda")
for path in [item for item in images if item.endswith(('-3.png', '-4.png'))]:
    from PIL import Image

    image = remover(Image.open(path).convert("RGB"))
    mesh = shape(image=image, num_inference_steps=30, octree_resolution=256, generator=torch.manual_seed(0))[0]
    mesh.export(path.replace(".png", ".glb"))
    print("MALHA", path.replace(".png", ".glb"), len(mesh.vertices), flush=True)
