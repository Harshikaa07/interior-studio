import numpy as np
from . import config


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class StyleClassifier:
    """Zero-shot CLIP by default; uses the trained head (train_style.py) if present."""

    def __init__(self, emb, use_head=True):
        import joblib
        self.names = list(config.STYLES)
        self.text = emb.texts([config.style_prompt(s) for s in self.names])
        f = config.INDEX / "style_head.joblib"
        self.clf = joblib.load(f)["clf"] if (use_head and f.exists()) else None

    def predict_matrix(self, E):
        if self.clf is None:
            return _softmax(100 * E @ self.text.T)
        p = np.zeros((len(E), len(self.names)))
        pp = self.clf.predict_proba(E)
        for j, c in enumerate(self.clf.classes_):
            p[:, self.names.index(c)] = pp[:, j]
        return p

    def predict(self, vec):
        return dict(zip(self.names, self.predict_matrix(vec[None])[0].tolist()))
