import numpy as np
from PIL import Image, ImageEnhance
from . import config

# used when no GPU is available: a quick style colour-grade so the app still works
GRADE = {
    "Modern": (0.85, 1.05, 1.10, (1.00, 1.00, 1.03)),
    "Scandinavian": (0.80, 1.12, 0.95, (1.02, 1.02, 1.04)),
    "Minimalist": (0.55, 1.10, 1.00, (1.00, 1.00, 1.00)),
    "Industrial": (0.70, 0.90, 1.20, (0.98, 0.97, 0.97)),
    "Bohemian": (1.35, 1.00, 1.05, (1.05, 1.00, 0.95)),
    "Rustic": (1.00, 0.95, 1.05, (1.07, 1.01, 0.92)),
    "Mid-century": (1.10, 1.00, 1.05, (1.06, 1.02, 0.94)),
    "Traditional": (0.95, 0.92, 1.05, (1.05, 1.00, 0.95)),
    "Coastal": (0.90, 1.12, 0.95, (0.96, 1.02, 1.07)),
    "Contemporary": (1.10, 1.00, 1.15, (1.00, 1.00, 1.00)),
}
NEG = "blurry, distorted, low quality, deformed furniture, text, watermark"


def grade(img, style):
    sat, bri, con, tint = GRADE[style]
    img = ImageEnhance.Contrast(ImageEnhance.Brightness(ImageEnhance.Color(img.convert("RGB")).enhance(sat)).enhance(bri)).enhance(con)
    return Image.fromarray(np.clip(np.asarray(img, float) * np.array(tint), 0, 255).astype("uint8"))


class Redesigner:
    """Stable Diffusion + ControlNet (canny edges keep walls/windows/layout in place).
    Falls back to a colour-grade preview when there is no CUDA GPU."""

    def __init__(self):
        self.pipe, self.mode = None, "preview"
        try:
            import torch
            if torch.cuda.is_available():
                from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                                       UniPCMultistepScheduler)
                cn = ControlNetModel.from_pretrained(config.CN_MODEL, torch_dtype=torch.float16)
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    config.SD_MODEL, controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                self.pipe, self.mode = pipe.to("cuda"), "diffusion"
        except Exception as e:  # missing libs / weights -> preview mode
            print("Redesigner: using preview mode:", e)

    def run(self, img, style):
        img = img.convert("RGB")
        if self.pipe is None:
            return grade(img, style)
        import cv2
        w, h = img.size
        s = 512 / max(w, h)
        w, h = int(w * s) // 8 * 8, int(h * s) // 8 * 8
        small = img.resize((w, h))
        edges = cv2.Canny(np.asarray(small), 100, 200)
        ctrl = Image.fromarray(np.stack([edges] * 3, -1))
        prompt = config.style_prompt(style) + ", same room layout, high quality"
        return self.pipe(prompt, negative_prompt=NEG, image=ctrl, num_inference_steps=25,
                         guidance_scale=7.5).images[0]
