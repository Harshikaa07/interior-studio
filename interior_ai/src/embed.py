import numpy as np
from PIL import Image
from . import config


def _feat(out):
    import torch
    if torch.is_tensor(out):
        return out
    return out.pooler_output if hasattr(out, "pooler_output") else out[0]


class Embedder:
    """CLIP: images and text share one space, so rooms, products and style words
    can all be compared with cosine similarity."""

    def __init__(self, model_name=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = model_name or config.CLIP_MODEL
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(name).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def _norm(self, f):
        f = f.float().cpu().numpy()
        return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype("float32")

    def images(self, imgs, bs=32):
        out = []
        for i in range(0, len(imgs), bs):
            x = self.proc(images=[im.convert("RGB") for im in imgs[i:i + bs]],
                          return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                out.append(self._norm(_feat(self.model.get_image_features(**x))))
        return np.concatenate(out) if out else np.zeros((0, 512), "float32")

    def texts(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            x = self.proc(text=list(texts[i:i + bs]), return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(self.dev)
            with self.torch.no_grad():
                out.append(self._norm(_feat(self.model.get_text_features(**x))))
        return np.concatenate(out) if out else np.zeros((0, 512), "float32")

    def paths(self, paths, bs=32):
        """Embed image files; unreadable ones are skipped. Returns (embeddings, ok_indices)."""
        from tqdm import tqdm
        embs, ok = [], []
        for i in tqdm(range(0, len(paths), bs), desc="embedding images"):
            imgs, idx = [], []
            for j, p in enumerate(paths[i:i + bs]):
                try:
                    imgs.append(Image.open(p).convert("RGB")); idx.append(i + j)
                except Exception:
                    pass
            if imgs:
                embs.append(self.images(imgs, bs)); ok += idx
        return (np.concatenate(embs) if embs else np.zeros((0, 512), "float32")), ok
