"""python datasets.py status   -> which datasets are present in data/
   python datasets.py fetch    -> download what can be automated, print steps for the rest
   python datasets.py kaggle owner/dataset styles|ikea|houzz  -> download a Kaggle dataset into data/"""
import shutil, subprocess, sys
from src import config

SOURCES = [
    dict(key="styles", name="Kaggle interior design styles", role="trains the style classifier", auto=None,
         how="Search Kaggle for 'interior design styles' (e.g. natecat/interior-design-styles or "
             "meherunnesashoshi/houzz-interior-design-dataset), then run:\n"
             "  python datasets.py kaggle <owner/dataset> styles\n"
             "Or unzip into data/styles/<style_name>/*.jpg. Folder names containing a style name "
             "(modern, scandinavian, ...) are detected automatically."),
    dict(key="houzz", name="Houzz Interior Design (Kaggle)", role="enriched style training data with 15 styles", auto=None,
         how="Search Kaggle for 'meherunnesashoshi/houzz-interior-design-dataset', then run:\n"
             "  python datasets.py kaggle meherunnesashoshi/houzz-interior-design-dataset houzz\n"
             "Images in data/houzz/ are auto-merged into the style pool."),
    dict(key="places365", name="Places365 (indoor classes)", role="room photos for style pseudo-labelling", auto="places365",
         how="automatic (torchvision, validation split, indoor classes only)"),
    dict(key="lsun", name="LSUN (bedroom / living room)", role="room photos for style pseudo-labelling", auto=None,
         how="git clone https://github.com/fyu/lsun, run download.py for bedroom and living_room, export ~1000 JPGs into data/lsun/"),
    dict(key="ade20k", name="ADE20K", role="trained the SegFormer used for object detection; its photos join the room pool", auto=None,
         how="download from https://ade20k.csail.mit.edu (or Hugging Face 'scene_parse_150'), put images in data/ade20k/"),
    dict(key="sun_rgbd", name="SUN RGB-D", role="indoor scene photos for the room pool", auto=None,
         how="download from https://rgbd.cs.princeton.edu, put RGB images in data/sun_rgbd/"),
    dict(key="abo", name="Amazon Berkeley Objects", role="product catalogue (photos + names + colours)", auto="abo",
         how="automatic: downloads the listings (~85 MB) and image index; build_index.py then fetches only the ~3000 photos it uses"),
    dict(key="ikea", name="IKEA furniture (Kaggle)", role="product catalogue (names, categories, prices)", auto=None,
         how="Search Kaggle for 'IKEA furniture dataset' (a CSV with item_id, name, category, price), then either:\n"
             "  python datasets.py kaggle <owner/dataset> ikea\n"
             "or unzip the CSV into data/ikea/"),
    dict(key="wands", name="Wayfair WANDS", role="product catalogue (names, categories)", auto="git:https://github.com/wayfair/WANDS",
         how="automatic (git clone)"),
    dict(key="stl", name="Pinterest Shop The Look", role="evaluation: scene -> product retrieval", auto="git:https://github.com/kang205/STL-Dataset",
         how="automatic (git clone of the home + fashion scene-product pairs); evaluate.py downloads the images"),
    dict(key="3dfront", name="3D-FRONT (optional, gated)", role="synthetic room renders for the room pool", auto=None,
         how="request access at https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset; put renders in data/3dfront/"),
    dict(key="structured3d", name="Structured3D (optional, gated)", role="synthetic room renders for the room pool", auto=None,
         how="request access at https://structured3d-dataset.org; put perspective renders in data/structured3d/"),
    dict(key="matterport3d", name="Matterport3D (optional, gated)", role="real room scans for the room pool", auto=None,
         how="sign the terms at https://niessner.github.io/Matterport/; put RGB images in data/matterport3d/"),
]


def present(key):
    d = config.DATA / key
    return d.exists() and any(d.rglob("*"))


def status():
    print(f"\n{'Dataset':<36}  {'Status':<8}  Role")
    print("-" * 80)
    for s in SOURCES:
        tick = "YES" if present(s["key"]) else "---"
        print(f"[{tick}] {s['name']:<34}  {s['role']}")
    print()


def places365():
    from torchvision.datasets import Places365
    indoor = {"living_room", "bedroom", "dining_room", "kitchen", "home_office", "childs_room", "television_room"}
    root = config.DATA / "_places_raw"
    ds = Places365(str(root), split="val", small=True, download=True)
    out = config.DATA / "places365"; out.mkdir(exist_ok=True)
    keep = {i for i, c in enumerate(ds.classes) if c.split("/")[-1] in indoor}
    n = 0
    for path, idx in ds.imgs:
        if idx in keep:
            shutil.copy(path, out / f"{idx}_{n}.jpg"); n += 1
    print(f"Places365: copied {n} indoor images")


def _get(url, dest):
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def abo():
    import tarfile
    base = "https://amazon-berkeley-objects.s3.amazonaws.com/"
    root = config.DATA / "abo"
    tar = root / "abo-listings.tar"
    _get(base + "archives/abo-listings.tar", tar)
    with tarfile.open(tar) as t:
        t.extractall(root)
    tar.unlink()
    _get(base + "images/metadata/images.csv.gz", root / "images" / "metadata" / "images.csv.gz")
    print("ABO listings + image index ready")


def kaggle(slug, target):
    """python datasets.py kaggle owner/dataset styles|ikea|houzz   (needs `pip install kaggle` and ~/.kaggle/kaggle.json)"""
    dest = config.DATA / target
    dest.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
    print("downloaded to", dest)


def fetch():
    for s in SOURCES:
        if present(s["key"]):
            continue
        a = s["auto"]
        print(f"\n== {s['name']}")
        if a and a.startswith("git:"):
            subprocess.run(["git", "clone", "--depth", "1", a[4:], str(config.DATA / s["key"])])
        elif a == "places365":
            try: places365()
            except Exception as e: print("failed:", e)
        elif a == "abo":
            try: abo()
            except Exception as e: print("failed:", e)
        else:
            print("manual:", s["how"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "kaggle" and len(sys.argv) == 4:
        kaggle(sys.argv[2], sys.argv[3])
    else:
        {"status": status, "fetch": fetch}.get(cmd, status)()
