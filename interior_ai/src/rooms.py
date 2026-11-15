"""Room-photo pools from the scene datasets (Places365, LSUN, ADE20K, SUN RGB-D and, if you
have access, Structured3D / 3D-FRONT / Matterport3D renders) + the Kaggle style folders."""
import random, re
from . import config

POOLS = ["places365", "lsun", "ade20k", "sun_rgbd", "structured3d", "3dfront", "matterport3d"]


def _files(d, limit, seed=0):
    fs = [p for p in d.rglob("*") if p.suffix.lower() in config.IMG_EXT]
    random.Random(seed).shuffle(fs)
    return fs[:limit]


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def labeled_style_images(per_class=400):
    """Style labels come from sub-folder names (data/styles/<style>/*.jpg), from file names
    (e.g. modern_01.jpg directly in data/styles), or from data/styles/style_map.json
    ({"folder name": "Style"}) for folders that don't contain a style name."""
    import json
    root, paths, labels, skipped = config.DATA / "styles", [], [], []
    names = sorted(config.STYLES, key=len, reverse=True)
    if not root.exists():
        return paths, labels, skipped
    mp = root / "style_map.json"
    manual = json.loads(mp.read_text()) if mp.exists() else {}
    find = lambda text: next((s for s in names if _norm(s) in _norm(text)), None)
    for d in sorted(root.iterdir()):
        if d.is_dir():
            style = manual.get(d.name) or find(d.name)
            if style not in config.STYLES:
                skipped.append(d.name); continue
            fs = _files(d, per_class)
            paths += fs; labels += [style] * len(fs)
    by_style = {}
    for f in root.iterdir():          # images sitting directly in data/styles: label from the file name
        if f.is_file() and f.suffix.lower() in config.IMG_EXT and find(f.stem):
            by_style.setdefault(find(f.stem), []).append(f)
    for style, fs in by_style.items():
        paths += fs[:per_class]; labels += [style] * len(fs[:per_class])
    return paths, labels, skipped


def unlabeled_room_images(per_pool=800):
    out = {}
    for p in POOLS:
        d = config.DATA / p
        if d.exists():
            fs = _files(d, per_pool)
            if fs:
                out[p] = fs
    return out
