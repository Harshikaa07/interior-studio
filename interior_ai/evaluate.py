"""python evaluate.py stl [N]
Scene -> product retrieval on Pinterest Shop The Look (home subset): given a room photo, how often does the
true product rank in the top k among N candidate products? Prints recall@1/5/10 vs a random baseline."""
import json, random, sys
import numpy as np, requests
from PIL import Image
from src import config



def load_pairs(limit, seed=0):
    root = config.DATA / "stl"
    f = next(iter(root.rglob("home.json")), None)
    if f is None:
        sys.exit("data/stl/home.json not found. Run: python datasets.py fetch")
    cats = {}
    cf = next(iter(root.rglob("home-cat.json")), None)
    if cf:
        cats = json.loads(cf.read_text())
    pairs = []
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if cats.get(r["product"], "").startswith("Hardware"):   # drop faucets/plumbing: not room furnishing
            continue
        pairs.append((r["scene"], r["product"]))
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


def _url(s, scheme="https"):
    return f"{scheme}://i.pinimg.com/400x/{s[:2]}/{s[2:4]}/{s[4:6]}/{s}.jpg"


def _get(sig):
    cache = config.DATA / "stl_cache"
    cache.mkdir(exist_ok=True)
    f = cache / f"{sig}.jpg"
    if not f.exists():
        for scheme in ("https", "http"):
            try:
                r = requests.get(_url(sig, scheme), timeout=15)
                r.raise_for_status()
                f.write_bytes(r.content)
                break
            except Exception:
                continue
    return Image.open(f).convert("RGB")


def stl(n=300):
    from tqdm import tqdm
    from src.embed import Embedder
    pairs = load_pairs(n)
    scenes, prods, ok = [], [], 0
    for s, p in tqdm(pairs, desc="downloading images"):
        try:
            a, b = _get(s), _get(p)
        except Exception:
            continue
        scenes.append(a); prods.append(b); ok += 1
    print(f"{ok} of {len(pairs)} pairs downloaded")
    if ok < 30:
        sys.exit("Too few images could be downloaded (these are 2019 Pinterest links and many are dead). "
                 "Try a larger N, e.g. python evaluate.py stl 2000")
    emb = Embedder()
    S, P = emb.images(scenes), emb.images(prods)
    sims = S @ P.T
    rank = (sims > np.diag(sims)[:, None]).sum(1)      # 0 = true product ranked first
    res = {f"recall@{k}": round(float((rank < k).mean()), 4) for k in (1, 5, 10)}
    res.update(n_pairs=ok, random_recall_at_10=round(min(1.0, 10 / ok), 4))
    print(json.dumps(res, indent=2))
    (config.INDEX / "stl_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["stl"]:
        stl(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        print(__doc__)
