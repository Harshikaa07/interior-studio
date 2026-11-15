import numpy as np
from . import config
from .palette import harmony_many, rgb_to_lab


def _mm(x):
    x = np.asarray(x, float)
    r = x.max() - x.min()
    return np.zeros_like(x) if r < 1e-9 else (x - x.min()) / r


class Recommender:
    """Hybrid content-based ranking:
    room look (CLIP similarity) + target style (CLIP text) + colour harmony + user taste."""
    W = dict(room=0.30, style=0.30, color=0.20, taste=0.20)

    def __init__(self, emb, df, E):
        self.emb, self.df, self.E = emb, df.reset_index(drop=True), E
        self.pos = {str(i): n for n, i in enumerate(self.df["id"])}
        self.has_rgb = self.df["rgb"].map(lambda c: c is not None).values
        self.lab = np.zeros((len(self.df), 3))
        if self.has_rgb.any():
            self.lab[self.has_rgb] = rgb_to_lab(np.array([c for c in self.df["rgb"] if c is not None], float))
        self._sv = {}

    def _style_vec(self, s):
        if s not in self._sv:
            self._sv[s] = self.emb.texts([config.style_prompt(s)])[0]
        return self._sv[s]

    def recommend(self, room_emb, style, palette, missing=(), budget=None, liked=(),
                  disliked=(), k=8, per_cat=2):
        df, n = self.df, len(self.df)
        room = _mm(self.E @ room_emb)
        sty = _mm(self.E @ self._style_vec(style))
        col = np.full(n, 0.5)
        if self.has_rgb.any():
            col[self.has_rgb] = harmony_many(palette, self.lab[self.has_rgb])
        w, taste = dict(self.W), np.zeros(n)
        li = [self.pos[str(i)] for i in liked if str(i) in self.pos]
        di = [self.pos[str(i)] for i in disliked if str(i) in self.pos]
        if li or di:
            v = np.zeros(self.E.shape[1])
            if li: v += self.E[li].mean(0)
            if di: v -= 0.5 * self.E[di].mean(0)
            taste = _mm(self.E @ v)
        else:
            w["taste"] = 0
        score = (w["room"] * room + w["style"] * sty + w["color"] * col + w["taste"] * taste) / sum(w.values())
        gap = df["category"].isin(list(missing)).values
        score = score + 0.08 * gap
        ok = np.ones(n, bool)
        if budget:
            ok &= (df["price"].isna() | (df["price"] <= budget)).values
        if di:
            ok[di] = False
        picks, seen = [], {}
        for i in np.argsort(-score):
            if not ok[i]:
                continue
            c = df.at[i, "category"]
            if seen.get(c, 0) >= per_cat:
                continue
            seen[c] = seen.get(c, 0) + 1
            why = []
            if sty[i] > 0.7: why.append(f"fits {style} style")
            if col[i] > 0.7: why.append("colour suits your room")
            if gap[i]: why.append(f"your room has no {c} yet")
            if room[i] > 0.7: why.append("similar look to your room")
            if taste[i] > 0.7 and li: why.append("like items you liked")
            picks.append((i, float(score[i]), " · ".join(why) or "good overall match"))
            if len(picks) >= k:
                break
        out = df.iloc[[p[0] for p in picks]].copy()
        out["score"], out["why"] = [p[1] for p in picks], [p[2] for p in picks]
        return out
