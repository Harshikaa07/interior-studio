import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    """RGB (0-255), shape (3,) or (N,3) -> CIE Lab."""
    c = np.asarray(rgb, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (c @ _M.T) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def dominant_colors(img, k=5):
    """K-means on pixels -> [((r,g,b), share), ...] sorted by share."""
    px = np.asarray(img.convert("RGB").resize((96, 96))).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(px)
    share = np.bincount(km.labels_, minlength=k) / len(px)
    return [(tuple(int(v) for v in km.cluster_centers_[i]), float(share[i]))
            for i in np.argsort(-share)]


def product_color(img):
    """Mean colour of a product photo, ignoring near-white background."""
    px = np.asarray(img.convert("RGB").resize((64, 64))).reshape(-1, 3).astype(float)
    fg = px[(px < 235).any(axis=1)]
    fg = fg if len(fg) > 20 else px
    return tuple(int(v) for v in fg.mean(0))


def harmony_many(palette, labs):
    """0..1 score per product Lab colour: close to a room colour (weighted by its
    share) or neutral enough to go with anything."""
    pl = rgb_to_lab([c for c, _ in palette])
    w = np.array([s for _, s in palette])
    d = np.linalg.norm(labs[:, None, :] - pl[None, :, :], axis=2)
    close = (np.exp(-d / 25) * (0.5 + w)).max(1)
    neutral = np.exp(-np.hypot(labs[:, 1], labs[:, 2]) / 12) * 0.6
    return np.clip(np.maximum(close, neutral), 0, 1)
