"""Logic tests with a fake CLIP/segmenter (no model downloads needed): python tests_core.py"""
import sys, types, zlib, numpy as np
from PIL import Image

class FakeEmb:
    def _v(self, seed):
        v = np.random.default_rng(seed).normal(size=64); return (v / np.linalg.norm(v)).astype("float32")
    def texts(self, t): return np.stack([self._v(zlib.crc32(x.encode())) for x in t])
    def images(self, ims): return np.stack([self._v(int(np.asarray(i).mean())) for i in ims])
    def paths(self, ps):
        ims = [Image.open(p) for p in ps]; return self.images(ims), list(range(len(ps)))
class FakeSeg:
    def objects(self, img): return {"wall": .4, "sofa": .2, "floor": .2, "lamp": .01}

import gzip, json, tempfile, pathlib
from src import config, catalog

# ---- dataset loaders on files written in each dataset's documented format ----
real_data = config.DATA
with tempfile.TemporaryDirectory() as tmp:
    config.DATA = pathlib.Path(tmp)
    abo = config.DATA / "abo"; (abo / "listings/metadata").mkdir(parents=True); (abo / "images/metadata").mkdir(parents=True)
    with gzip.open(abo / "listings/metadata/listings_0.json.gz", "wt") as f:
        for i, (n, t, col) in enumerate([("Modern Linen Sofa", "SOFA", "Grey"), ("Wooden Dining Chair", "CHAIR", "Brown"), ("USB Cable", "CABLE", "Black")]):
            f.write(json.dumps({"item_id": f"B{i}", "item_name": [{"language_tag": "de_DE", "value": "x"}, {"language_tag": "en_US", "value": n}],
                                "product_type": [{"value": t}], "color": [{"language_tag": "en_US", "value": col}], "main_image_id": f"img{i}"}) + "\n")
    (abo / "images/metadata/images.csv").write_text("image_id,height,width,path\nimg0,1,1,aa/a.jpg\nimg1,1,1,bb/b.jpg\n")
    rows = catalog.load_abo(download_images=False)
    assert {r["category"] for r in rows} == {"sofa", "chair"} and len(rows) == 2          # cable filtered out
    assert [r for r in rows if r["category"] == "sofa"][0]["rgb"] == config.COLOR_WORDS["grey"]
    assert all(r["image_path"].endswith(".jpg") and "images" in r["image_path"] for r in rows)

    ik = config.DATA / "ikea"; ik.mkdir()
    (ik / "ikea.csv").write_text("item_id,name,category,price,short_description\n1.0,BILLY,Bookcases,100,White bookcase\n2.0,FREKVENS,Bar furniture,50,Grey stool\n")
    ir = catalog.load_ikea()
    assert abs(ir[0]["price"] - 100 * config.CURRENCY_TO_INR["SAR"]) < 1e-6 and ir[0]["id"] == "ikea:1"   # SAR -> INR
    (ik / "ikea.csv").write_text("item_id,name,category,price,currency\n3,KALLAX,Shelf unit,10,USD\n")
    assert abs(catalog.load_ikea()[0]["price"] - 10 * config.CURRENCY_TO_INR["USD"]) < 1e-6          # currency column respected

    st = config.DATA / "styles"; (st / "Scandinavian_living").mkdir(parents=True); (st / "weird_folder").mkdir()
    for pth in [st / "Scandinavian_living/a.jpg", st / "weird_folder/b.jpg", st / "modern_01.jpg"]:
        Image.new("RGB", (8, 8)).save(pth)
    from src.rooms import labeled_style_images
    ps, ls, skipped = labeled_style_images()
    assert sorted(ls) == ["Modern", "Scandinavian"] and skipped == ["weird_folder"]
    (st / "style_map.json").write_text(json.dumps({"weird_folder": "Industrial"}))
    assert "Industrial" in labeled_style_images()[1]
config.DATA = real_data
print("loaders OK (ABO, IKEA+INR, style labelling)")

assert config.categorize("floor lamp") == "lighting" and config.categorize("Coffee table") == "table"
assert config.categorize("bedside table") == "table" and config.categorize("banana") is None

import build_index
build_index.build(FakeEmb())
df, E = catalog.load_index()
print("catalog:", len(df), "items, emb", E.shape)

from src.recommend import Recommender
from src.generate import Redesigner
from src.pipeline import analyze_room
from src.palette import dominant_colors
class FakeStyle:
    def predict(self, v): return {s: 1 / 10 for s in config.STYLES} | {"Coastal": .5}
emb = FakeEmb(); rec = Recommender(emb, df, E)
m = dict(emb=emb, style=FakeStyle(), seg=FakeSeg(), redesign=Redesigner(), rec=rec)
img = Image.fromarray((np.random.default_rng(0).random((200, 300, 3)) * 255).astype("uint8"))
res = analyze_room(img, m, "Scandinavian")
assert res["redesign"].size == img.size and res["target"] == "Scandinavian"
assert "sofa" in res["have"] and "rug" in res["missing"] and "sofa" not in res["missing"]
r = rec.recommend(res["room_emb"], "Scandinavian", res["palette"], res["missing"], budget=30000)
assert len(r) == 8 and (r.price <= 30000).all() and r.category.value_counts().max() <= 2
r2 = rec.recommend(res["room_emb"], "Scandinavian", res["palette"], res["missing"], liked=[r.id.iloc[0]], disliked=[r.id.iloc[1]])
assert r.id.iloc[1] not in set(r2.id)
print(r[["name", "category", "price", "why"]].head(4).to_string())
print("core logic OK, redesign mode:", m["redesign"].mode)

# UI smoke test: app renders its empty state with models faked
for name, cls in dict(embed="Embedder", style="StyleClassifier", segment="Segmenter").items():
    mod = types.ModuleType(f"src.{name}")
    setattr(mod, cls, {"Embedder": FakeEmb, "StyleClassifier": lambda e: FakeStyle(), "Segmenter": FakeSeg}[cls])
    sys.modules[f"src.{name}"] = mod
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
assert (df.price >= 2000).all(); print("app renders:", [t.value for t in at.title], "| button disabled until upload:", at.button[0].disabled)
