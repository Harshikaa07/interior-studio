"""Train the style classifier.
 - labelled data: Kaggle interior-style folders (data/styles/<style>/*.jpg)
 - extra data: room photos from Places365 / LSUN / ADE20K / SUN RGB-D / ... are pseudo-labelled by
   zero-shot CLIP (only confident, indoor ones kept) and added at half weight
 - model: logistic regression on frozen CLIP embeddings; compared with the zero-shot baseline."""
import json, sys
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src import config
from src.embed import Embedder
from src.rooms import labeled_style_images, unlabeled_room_images
from src.style import StyleClassifier


def main():
    emb = Embedder()
    zs = StyleClassifier(emb, use_head=False)
    paths, labels, skipped = labeled_style_images()
    if skipped:
        print("Folders not matched to a style (rename them to a style name):", skipped)
    if len(set(labels)) < 2:
        sys.exit("Put the Kaggle style images in data/styles/<style name>/*.jpg first (see README).")
    X, ok = emb.paths([str(p) for p in paths])
    y = np.array([labels[i] for i in ok])
    print(f"{len(y)} labelled images, {len(set(y))} styles")

    Xp, yp = np.zeros((0, X.shape[1]), "float32"), np.array([])
    pools = unlabeled_room_images()
    if pools:
        allp = [str(p) for fs in pools.values() for p in fs]
        E, _ = emb.paths(allp)
        room_txt = emb.texts(["a photo of the inside of a home", "an outdoor landscape or street photo"])
        z = 100 * E @ room_txt.T
        indoor = (np.exp(z[:, 0]) / np.exp(z).sum(1)) > 0.8
        P = zs.predict_matrix(E)
        keep = indoor & (P.max(1) > 0.5)
        Xp, yp = E[keep], np.array(zs.names)[P[keep].argmax(1)]
        print(f"pseudo-labelled {keep.sum()} of {len(E)} room photos from {list(pools)}")

    strat = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=0)
    base = (np.array(zs.names)[zs.predict_matrix(Xte).argmax(1)] == yte).mean()

    def fit(Xl, yl):
        clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
        Xa, ya = np.vstack([Xl, Xp]), np.concatenate([yl, yp])
        wa = np.concatenate([np.ones(len(yl)), np.full(len(yp), 0.5)])
        return clf.fit(Xa, ya, sample_weight=wa)

    acc = (fit(Xtr, ytr).predict(Xte) == yte).mean()
    print(f"held-out accuracy  zero-shot CLIP: {base:.3f}   trained head: {acc:.3f}")
    joblib.dump({"clf": fit(X, y)}, config.INDEX / "style_head.joblib")
    (config.INDEX / "style_metrics.json").write_text(json.dumps(
        dict(zero_shot_acc=float(base), trained_acc=float(acc), n_labelled=int(len(y)),
             n_pseudo=int(len(yp)), pools=list(pools)), indent=2))


if __name__ == "__main__":
    main()
