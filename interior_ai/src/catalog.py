"""Loads the product datasets (Amazon Berkeley Objects, IKEA, Wayfair WANDS) into one table:
id, source, name, category, price (INR), rgb, image_path, link, shop_link."""
import gzip, json, random, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from . import config
from .palette import product_color

COLS = ["id", "source", "name", "category", "price", "rgb", "image_path", "link", "shop_link"]
ABO_IMG_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"
WANDS_TOP = {"Furniture", "Decor & Pillows", "Lighting", "Rugs", "Storage & Organization",
             "Decor \u0026 Pillows"}


def _rgb_from_text(text):
    t = str(text).lower()
    for w, c in config.COLOR_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return c
    return None


def _en(field):
    if not field:
        return ""
    for d in field:
        if str(d.get("language_tag", "")).startswith("en"):
            return d.get("value", "")
    return field[0].get("value", "")


def _download(url, dest):
    try:
        if dest.exists():
            return True
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def _make_shop_link(source, name, category, existing_link=None):
    """Generate a shopping link based on dataset source."""
    if existing_link:
        return str(existing_link)
    if source == "wands":
        return config.wayfair_link(name, category)
    if source == "abo":
        return config.amazon_link(name, category)
    if source == "ikea":
        return config.ikea_link(name, category)
    return config.google_shopping_link(name, category)


def load_abo(limit=3000, download_images=True):
    """Needs listings_*.json.gz and images.csv(.gz) in data/abo (datasets.py fetch gets them).
    Only the photos of the sampled products are downloaded (a few MB, not the 3 GB archive)."""
    root = config.DATA / "abo"
    if not root.exists():
        return []
    imgmap, base = {}, root / "images" / "small"
    csvs = list(root.rglob("images.csv*"))
    if csvs:
        m = pd.read_csv(csvs[0])
        imgmap = dict(zip(m["image_id"], m["path"]))
        if (csvs[0].parent.parent / "small").exists():
            base = csvs[0].parent.parent / "small"
    cands = []
    for f in sorted(root.rglob("listings_*.json*")):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf8") as fh:
            for line in fh:
                d = json.loads(line)
                name, ptype = _en(d.get("item_name")), _en(d.get("product_type"))
                cat = config.categorize(f"{name} {ptype}")
                if not (cat and name):
                    continue
                rel = imgmap.get(d.get("main_image_id"))
                link = _make_shop_link("abo", name, cat)
                cands.append(dict(id=f"abo:{d['item_id']}", source="abo", name=name[:120], category=cat,
                                  price=None, rgb=_rgb_from_text(_en(d.get("color"))),
                                  image_path=str(base / rel) if rel else None, link=None,
                                  shop_link=link, _rel=rel))
    random.Random(0).shuffle(cands)
    rows = cands[:limit]
    if download_images:
        todo = [(ABO_IMG_URL + r["_rel"], base / r["_rel"]) for r in rows if r["_rel"]]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda a: _download(*a), todo))
    for r in rows:
        r.pop("_rel")
    return rows


def load_ikea(limit=3000):
    root = config.DATA / "ikea"
    csvs = list(root.rglob("*.csv")) if root.exists() else []
    if not csvs:
        return []
    df = pd.read_csv(csvs[0])
    col = {c.lower(): c for c in df.columns}
    g = lambda r, k: str(r[col[k]]) if k in col and pd.notna(r[col[k]]) else ""
    imgs = {p.stem: str(p) for p in root.rglob("*") if p.suffix.lower() in config.IMG_EXT}
    rows = []
    for _, r in df.iterrows():
        name, cat_txt, desc = g(r, "name"), g(r, "category"), g(r, "short_description")
        cat = config.categorize(f"{name} {cat_txt} {desc}")
        if not cat:
            continue
        price = pd.to_numeric(g(r, "price"), errors="coerce")
        cur = (g(r, "currency") or config.IKEA_DEFAULT_CURRENCY).upper()
        inr = None if pd.isna(price) else float(price) * config.CURRENCY_TO_INR.get(cur, 1.0)
        iid = g(r, "item_id").split(".")[0] or str(len(rows))
        existing_link = g(r, "link") or None
        shop_link = existing_link or config.ikea_link(name, cat)
        rows.append(dict(id=f"ikea:{iid}", source="ikea", name=f"IKEA {name} {desc}".strip()[:120],
                         category=cat, price=inr, rgb=_rgb_from_text(f"{desc} {name}"),
                         image_path=imgs.get(iid), link=existing_link, shop_link=shop_link))
        if len(rows) >= limit:
            break
    return rows


def load_wands(limit=3000):
    root = config.DATA / "wands"
    f = next(iter(root.rglob("product.csv")), None) if root.exists() else None
    if f is None:
        return []
    df = pd.read_csv(f, sep="\t")
    hier = df["category hierarchy"].fillna("").str.split(" / ").str[0]
    df = df[hier.isin(WANDS_TOP)]            # indoor furniture, decor, lighting, rugs, storage
    rows = []
    for r in df.sample(frac=1, random_state=0).itertuples():
        name, cls = str(r.product_name), str(getattr(r, "product_class", ""))
        cat = config.categorize(cls) or config.categorize(name)
        if not cat:
            continue
        shop_link = config.wayfair_link(name, cat)
        rows.append(dict(id=f"wands:{r.product_id}", source="wands", name=name[:120], category=cat,
                         price=None, rgb=_rgb_from_text(name), image_path=None, link=None,
                         shop_link=shop_link))
        if len(rows) >= limit:
            break
    return rows


def demo_catalog():
    """Tiny synthetic catalogue so the UI runs before any dataset is downloaded."""
    d = config.DATA / "demo"
    d.mkdir(exist_ok=True)
    names = {"sofa":    ["Linen sofa", "Velvet sectional", "Leather couch"],
             "chair":   ["Accent armchair", "Wooden dining chair", "Lounge chair"],
             "table":   ["Coffee table", "Dining table", "Side table"],
             "lighting":["Floor lamp", "Pendant light", "Table lamp"],
             "rug":     ["Wool rug", "Jute rug", "Patterned rug"],
             "curtain": ["Linen curtains", "Blackout curtains"],
             "storage": ["Bookcase", "Sideboard", "Wardrobe"],
             "decor":   ["Wall mirror", "Ceramic vase", "Cushion set"]}
    cols, rng, rows = list(config.COLOR_WORDS)[:10], random.Random(0), []
    for cat, ns in names.items():
        for n in ns:
            for cn in rng.sample(cols, 3):
                rgb = config.COLOR_WORDS[cn]
                p = d / f"{cat}_{len(rows)}.png"
                im = Image.new("RGB", (240, 240), (245, 245, 245))
                ImageDraw.Draw(im).rounded_rectangle((40, 60, 200, 190), 18, fill=rgb)
                im.save(p)
                shop_link = config.google_shopping_link(f"{cn.title()} {n}", cat)
                rows.append(dict(id=f"demo:{len(rows)}", source="demo", name=f"{cn.title()} {n}",
                                 category=cat, price=float(rng.randrange(2000, 80000, 500)), rgb=rgb,
                                 image_path=str(p), link=None, shop_link=shop_link))
    return rows


def build_all():
    parts = {"abo": load_abo(), "ikea": load_ikea(), "wands": load_wands()}
    for k, v in parts.items():
        print(f"{k}: {len(v)} products")
    rows = [r for v in parts.values() for r in v]
    if not rows:
        print("No product datasets found in data/ -> using the small demo catalogue.")
        rows = demo_catalog()
    df = pd.DataFrame(rows, columns=COLS)
    for i, r in df.iterrows():   # colour from the photo where the datasets gave none
        if r.rgb is None and r.image_path and re.search(r"\.(jpe?g|png|webp)$", str(r.image_path), re.I):
            try:
                df.at[i, "rgb"] = product_color(Image.open(r.image_path))
            except Exception:
                pass
    return df


def save(df, E):
    out = df.copy()
    out["rgb"] = out["rgb"].map(lambda c: ",".join(map(str, c)) if c is not None else "")
    out.to_csv(config.INDEX / "catalog.csv", index=False)
    np.save(config.INDEX / "embeddings.npy", E)


def load_index():
    df = pd.read_csv(config.INDEX / "catalog.csv")
    df["rgb"] = df["rgb"].map(lambda s: tuple(int(x) for x in s.split(",")) if isinstance(s, str) and s else None)
    df["image_path"] = df["image_path"].where(df["image_path"].notna(), None)
    df["shop_link"] = df["shop_link"].where(df["shop_link"].notna(), None)
    return df, np.load(config.INDEX / "embeddings.npy")
