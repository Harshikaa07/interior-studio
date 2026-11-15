from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from src import config

st.set_page_config(page_title="Interior Studio", page_icon="🛋️", layout="wide")
st.markdown("""<style>
.block-container{max-width:1080px;padding-top:2rem}
h1{font-weight:600;letter-spacing:-.02em;margin-bottom:0}
.sw{display:inline-block;width:40px;height:40px;border-radius:8px;margin:0 6px 4px 0;border:1px solid #8884}
.why{opacity:.65;font-size:.82rem;line-height:1.3;min-height:2.2em}
</style>""", unsafe_allow_html=True)

if not (config.INDEX / "embeddings.npy").exists():
    st.title("Interior Studio")
    st.error("The product index isn't built yet. Run `python build_index.py` once, then reload this page.")
    st.stop()


@st.cache_resource(show_spinner="Loading models (first run takes a while)…")
def load_models():
    from src.catalog import load_index
    from src.embed import Embedder
    from src.generate import Redesigner
    from src.recommend import Recommender
    from src.segment import Segmenter
    from src.style import StyleClassifier
    emb = Embedder()
    df, E = load_index()
    return dict(emb=emb, style=StyleClassifier(emb), seg=Segmenter(),
                redesign=Redesigner(), rec=Recommender(emb, df, E))


models = load_models()
ss = st.session_state
ss.setdefault("res", None); ss.setdefault("liked", []); ss.setdefault("disliked", [])


def vote(kind, item_id):
    ss[kind].append(str(item_id))


st.title("Interior Studio")
st.caption("Upload a photo of your room, choose a style, and get a redesign with furniture that fits.")

left, right = st.columns(2, gap="large")
with left:
    up = st.file_uploader("Room photo", type=["jpg", "jpeg", "png", "webp"])
    target = st.selectbox("Style", ["Same as detected"] + list(config.STYLES))
    budget = st.slider("Max price per item (₹, 0 = no limit)", 0, 200000, 0, step=5000)
    go = st.button("Redesign my room", type="primary", disabled=up is None)
with right:
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Your room")

if go and up:
    from src.pipeline import analyze_room
    with st.spinner("Analysing and redesigning…"):
        ss.res = analyze_room(img, models, target)
        ss.liked, ss.disliked = [], []

res = ss.res
if res:
    t_redesign, t_recs, t_room = st.tabs(["Redesign", "Recommendations", "Room analysis"])

    with t_redesign:
        a, b = st.columns(2)
        a.image(res["original"], caption="Before")
        b.image(res["redesign"], caption=f"After: {res['target']}")
        if models["redesign"].mode == "preview":
            st.info("No GPU found, so this is a quick style colour-grade. Run on a GPU (e.g. Google Colab) for the full AI redesign.")

    with t_recs:
        recs = models["rec"].recommend(res["room_emb"], res["target"], res["palette"], res["missing"],
                                       budget or None, ss.liked, ss.disliked)
        st.caption("Tap 👍 or 👎 and the list adapts to your taste.")
        for row in range(0, len(recs), 4):
            cols = st.columns(4)
            for col, (_, r) in zip(cols, recs.iloc[row:row + 4].iterrows()):
                with col.container(border=True):
                    if r.image_path and Path(str(r.image_path)).exists():
                        st.image(r.image_path)
                    else:
                        st.markdown(f"<div style='height:120px;display:flex;align-items:center;justify-content:center;opacity:.5'>{r.category}</div>", unsafe_allow_html=True)
                    st.markdown(f"**{r['name'][:48]}**")
                    price = config.rupees(r.price) if pd.notna(r.price) else "price n/a"
                    st.caption(f"{r.category} · {price}")
                    st.markdown(f"<div class='why'>{r.why}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.button("👍", key=f"l{r['id']}", on_click=vote, args=("liked", r["id"]))
                    c2.button("👎", key=f"d{r['id']}", on_click=vote, args=("disliked", r["id"]))

    with t_room:
        st.markdown(f"**Detected style:** {res['detected']}")
        top = pd.Series(res["probs"]).sort_values(ascending=False).head(5)
        st.bar_chart(top)
        st.markdown("**Colour palette**")
        st.markdown("".join(f"<span class='sw' style='background:rgb{c}' title='{s:.0%}'></span>"
                            for c, s in res["palette"]), unsafe_allow_html=True)
        st.markdown("**Found in your room:** " + (", ".join(sorted(res["have"])) or "nothing recognised"))
        st.markdown("**Missing:** " + (", ".join(res["missing"]) or "nothing, the room is well furnished"))
