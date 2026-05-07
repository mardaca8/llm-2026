from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
    BlipProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

IMAGES_DIR = Path(__file__).parent / "images"
IMAGE_EXTS = {".jpeg"}
N_IMAGES = 10

DEVICE = "mps"

def load_images() -> list[tuple[str, Image.Image]]:
    paths = sorted(p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    paths = paths[:N_IMAGES]

    items: list[tuple[str, Image.Image]] = []
    for p in paths:
        items.append((p.name, Image.open(p).convert("RGB")))
    print(f"[images] loaded {len(items)} local files from {IMAGES_DIR}")
    return items


class BLIPCaptioner:
    name = "Salesforce/blip-image-captioning-large"

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(self.name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.name).to(device).eval()

    @torch.no_grad()
    def caption(self, img: Image.Image) -> str:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=32, num_beams=4)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()


class ViTGPT2Captioner:
    name = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.name).to(device).eval()

    @torch.no_grad()
    def caption(self, img: Image.Image) -> str:
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixel_values, max_new_tokens=32, num_beams=4)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


class GITCaptioner:
    name = "microsoft/git-base-coco"

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name).to(device).eval()

    @torch.no_grad()
    def caption(self, img: Image.Image) -> str:
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixel_values=pixel_values, max_new_tokens=32, num_beams=4)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    images = load_images()
    if not images:
        print("error no images to caption", file=sys.stderr)
        sys.exit(1)

    print("init loading BLIP-large")
    cap0 = BLIPCaptioner()
    print("init loading ViT-GPT2")
    cap1 = ViTGPT2Captioner()
    print("init loading GIT-base-coco")
    cap2 = GITCaptioner()
    captioners = [cap0, cap1, cap2]

    from termcolor import colored

    print()
    for label, img in images:
        print(f"# {label}")
        for cap in captioners:
            try:
                text = cap.caption(img)
            except Exception as e:
                text = f"<error: {e}>"
            print(colored(f"  [{cap.name}] {text}", "green"))
        print()


if __name__ == "__main__":
    main()
