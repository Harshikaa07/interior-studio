import numpy as np
from . import config


class Segmenter:
    """SegFormer (ADE20K) -> which objects are in the room and how much area they cover."""

    def __init__(self):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.proc = SegformerImageProcessor.from_pretrained(config.SEG_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(config.SEG_MODEL).eval()

    def objects(self, img):
        torch = self.torch
        x = self.proc(images=img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**x).logits
        up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear",
                                             align_corners=False)
        lab = up.argmax(1)[0].numpy()
        ids, counts = np.unique(lab, return_counts=True)
        return {self.model.config.id2label[int(i)]: float(c) / lab.size for i, c in zip(ids, counts)}


def detected_categories(objs, min_frac=0.003):
    return {c for name, f in objs.items() if f >= min_frac and (c := config.categorize(name))}
