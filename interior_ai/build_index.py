"""Embed every product (photo if we have one, otherwise its text) with CLIP and save the index."""
from pathlib import Path
import numpy as np
from src import catalog


def build(emb=None):
    df = catalog.build_all()
    if emb is None:
        from src.embed import Embedder
        emb = Embedder()
    dim = emb.texts(["x"]).shape[1]
    E = np.zeros((len(df), dim), "float32")
    has_img = np.array([bool(p) and Path(str(p)).exists() and s != "demo"
                        for p, s in zip(df.image_path, df.source)])
    idx = np.where(has_img)[0]
    if len(idx):
        e, ok = emb.paths(df.image_path.iloc[idx].tolist())
        E[idx[ok]] = e
    todo = np.where(np.abs(E).sum(1) == 0)[0]
    if len(todo):
        txt = [f"{df.category.iloc[i]}: {df.name.iloc[i]}" for i in todo]
        E[todo] = emb.texts(txt)
    catalog.save(df, E)
    print(f"Indexed {len(df)} products ({len(idx)} by photo, {len(todo)} by text).")


if __name__ == "__main__":
    build()
