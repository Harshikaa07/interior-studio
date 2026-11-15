"""Interior Studio – AI room visualizer & furniture recommender.

Upload a photo of your room, choose an aesthetic style, get a redesigned room image
and curated furniture recommendations with shop links.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from src import config

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Interior Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STYLE_EMOJI = {
    "Modern": "◼",
    "Scandinavian": "❄",
    "Minimalist": "○",
    "Industrial": "⚙",
    "Bohemian": "✿",
    "Rustic": "🌿",
    "Mid-century": "◇",
    "Traditional": "⬡",
    "Coastal": "〜",
    "Contemporary": "△",
    "Art Deco": "✦",
    "Japandi": "☽",
}

STYLE_COLORS = {
    "Modern":       "#2D2D2D",
    "Scandinavian": "#7BA7BC",
    "Minimalist":   "#A8A89A",
    "Industrial":   "#6B6B6B",
    "Bohemian":     "#C4873B",
    "Rustic":       "#8B5E3C",
    "Mid-century":  "#D4793B",
    "Traditional":  "#7B5E3C",
    "Coastal":      "#4A90B8",
    "Contemporary": "#3D3D5C",
    "Art Deco":     "#C5A028",
    "Japandi":      "#8A8878",
}

st.markdown("""
<style>
/* ---- Base ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #FAFAF8;
}

/* ---- Header ---- */
.studio-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.studio-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 500;
    color: #1A1A1A;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.studio-header p {
    color: #6B6B6B;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ---- Style pill ---- */
.style-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: white;
    margin-bottom: 0.8rem;
}

/* ---- Product card ---- */
.product-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #EBEBEB;
    transition: all 0.2s ease;
    height: 100%;
}
.product-card:hover {
    border-color: #D0D0D0;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}
.product-card-body {
    padding: 1rem;
}
.product-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: #1A1A1A;
    margin-bottom: 0.3rem;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-price {
    font-size: 0.95rem;
    font-weight: 600;
    color: #2D7B45;
    margin-bottom: 0.3rem;
}
.product-category {
    font-size: 0.72rem;
    color: #9B9B9B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.product-why {
    font-size: 0.78rem;
    color: #7A7A7A;
    line-height: 1.4;
    font-style: italic;
    min-height: 2.2em;
}

/* ---- Shop button ---- */
.shop-btn {
    display: block;
    text-align: center;
    padding: 0.5rem 1rem;
    background: #1A1A1A;
    color: white !important;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    text-decoration: none;
    margin-top: 0.8rem;
    letter-spacing: 0.03em;
    transition: background 0.15s ease;
}
.shop-btn:hover {
    background: #3D3D3D;
    color: white !important;
}

/* ---- Colour swatch ---- */
.swatch {
    display: inline-block;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    margin: 0 5px 5px 0;
    border: 1.5px solid rgba(0,0,0,0.08);
    vertical-align: middle;
}

/* ---- Analysis panel ---- */
.analysis-row {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid #EBEBEB;
    margin-bottom: 0.8rem;
}
.analysis-label {
    font-size: 0.72rem;
    color: #9B9B9B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.analysis-value {
    font-size: 0.95rem;
    color: #1A1A1A;
    font-weight: 400;
}

/* ---- Section headings ---- */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #1A1A1A;
    margin: 1.5rem 0 1rem 0;
    font-weight: 400;
}

/* ---- Upload area ---- */
.upload-hint {
    background: white;
    border: 1.5px dashed #D0D0D0;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #9B9B9B;
    font-size: 0.9rem;
}

/* ---- GPU warning ---- */
.gpu-note {
    background: #FFF8E7;
    border: 1px solid #F0D080;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    font-size: 0.82rem;
    color: #7A5F20;
    margin-top: 0.8rem;
}

/* ---- Streamlit overrides ---- */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.stButton > button[kind="primary"] {
    background: #1A1A1A !important;
    border: none !important;
    color: white !important;
    padding: 0.6rem 2rem !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.88rem;
    font-weight: 500;
}
div[data-testid="column"] { gap: 0.75rem; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Guard: index must exist before loading models
# ---------------------------------------------------------------------------
if not (config.INDEX / "embeddings.npy").exists():
    st.markdown("""
    <div class="studio-header">
        <h1>Interior Studio</h1>
        <p>AI-powered room visualizer & furniture recommender</p>
    </div>
    """, unsafe_allow_html=True)
    st.error(
        "**Product index not built yet.**  "
        "Run `python build_index.py` once in your terminal, then refresh this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models — first run takes a few minutes...")
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
ss.setdefault("res", None)
ss.setdefault("liked", [])
ss.setdefault("disliked", [])


def vote(kind, item_id):
    ss[kind].append(str(item_id))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="studio-header">
    <h1>Interior Studio</h1>
    <p>Upload a photo of your room &nbsp;·&nbsp; choose an aesthetic &nbsp;·&nbsp; get a redesign & curated furniture picks</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    up = st.file_uploader(
        "Drop your room photo here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPEG, PNG and WebP.",
    )
    if not up:
        st.markdown("""
        <div class="upload-hint">
            📷  Upload a photo of your living room, bedroom, kitchen, or any interior space
        </div>
        """, unsafe_allow_html=True)

    style_opts = ["Auto-detect"] + list(config.STYLES)
    target = st.selectbox(
        "Aesthetic / style",
        style_opts,
        format_func=lambda s: f"{STYLE_EMOJI.get(s, '')} {s}" if s != "Auto-detect" else "🔍 Auto-detect from photo",
    )
    budget = st.slider(
        "Max price per item (Rs., 0 = no limit)",
        0, 200_000, 0, step=5_000,
        format="Rs.%d",
    )

    go = st.button(
        "Redesign my room →",
        type="primary",
        disabled=up is None,
        use_container_width=True,
    )

    if up and models["redesign"].mode == "preview":
        st.markdown("""
        <div class="gpu-note">
            ℹ No GPU detected — redesign uses a quick colour-grade preview.
            Run on Google Colab (T4 GPU) for the full AI redesign with Stable Diffusion.
        </div>
        """, unsafe_allow_html=True)

with right:
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Your room", use_container_width=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if go and up:
    from src.pipeline import analyze_room
    with st.spinner("Analysing your room and redesigning..."):
        ss.res = analyze_room(img, models, None if target == "Auto-detect" else target)
        ss.liked, ss.disliked = [], []

res = ss.res

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if res:
    t_redesign, t_recs, t_analysis = st.tabs(["✨ Redesign", "🛒 Recommendations", "🔍 Room Analysis"])

    # ---- Redesign tab ----
    with t_redesign:
        col_a, col_b = st.columns(2)
        col_a.image(res["original"], caption="Before", use_container_width=True)
        col_b.image(res["redesign"], caption=f"After — {res['target']}", use_container_width=True)

        # Style pill
        style_color = STYLE_COLORS.get(res["target"], "#555")
        style_desc = config.STYLES.get(res["target"], "")
        st.markdown(f"""
        <div style="margin-top:1rem">
            <span class="style-pill" style="background:{style_color}">
                {STYLE_EMOJI.get(res['target'], '')} {res['target']}
            </span>
            <span style="color:#6B6B6B;font-size:0.88rem;"> — {style_desc}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---- Recommendations tab ----
    with t_recs:
        recs = models["rec"].recommend(
            res["room_emb"], res["target"], res["palette"], res["missing"],
            budget or None, ss.liked, ss.disliked
        )

        st.markdown(f"""
        <p class="section-title">Furniture picks for your {res['target']} room</p>
        <p style="color:#6B6B6B;font-size:0.85rem;margin-bottom:1.2rem">
            Click 👍 or 👎 to refine the list to your taste.
        </p>
        """, unsafe_allow_html=True)

        for row_start in range(0, len(recs), 4):
            cols = st.columns(4, gap="small")
            for col, (_, r) in zip(cols, recs.iloc[row_start:row_start + 4].iterrows()):
                with col:
                    # Image / placeholder
                    if r.image_path and Path(str(r.image_path)).exists():
                        st.image(r.image_path, use_container_width=True)
                    else:
                        # Colour swatch placeholder
                        rgb = r.rgb if r.rgb else (200, 200, 195)
                        hex_col = "#{:02x}{:02x}{:02x}".format(*rgb)
                        st.markdown(f"""
                        <div style="
                            background:{hex_col};
                            height:140px;
                            border-radius:12px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            margin-bottom:0.5rem;
                            font-size:0.75rem;
                            color:rgba(0,0,0,0.4);
                        ">{r.category.title()}</div>
                        """, unsafe_allow_html=True)

                    # Name & price
                    price_str = config.rupees(r.price) if pd.notna(r.price) else "Price N/A"
                    source_badge = {"abo": "Amazon", "ikea": "IKEA", "wands": "Wayfair", "demo": "Demo"}.get(r.source, r.source.title())

                    st.markdown(f"""
                    <div class="product-category">{r.category.upper()} &nbsp;·&nbsp; {source_badge}</div>
                    <div class="product-name">{r['name'][:60]}</div>
                    <div class="product-price">{price_str}</div>
                    <div class="product-why">{r.why}</div>
                    """, unsafe_allow_html=True)

                    # Shop link
                    shop_url = r.get("shop_link") if hasattr(r, "get") else None
                    if shop_url and pd.notna(shop_url):
                        st.markdown(f'<a href="{shop_url}" target="_blank" class="shop-btn">Shop Now →</a>',
                                    unsafe_allow_html=True)

                    # Vote buttons
                    c1, c2 = st.columns(2)
                    c1.button("👍", key=f"l{r['id']}", on_click=vote, args=("liked", r["id"]),
                              use_container_width=True)
                    c2.button("👎", key=f"d{r['id']}", on_click=vote, args=("disliked", r["id"]),
                              use_container_width=True)

    # ---- Room analysis tab ----
    with t_analysis:
        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.markdown('<p class="section-title">Style breakdown</p>', unsafe_allow_html=True)
            prob_series = pd.Series(res["probs"]).sort_values(ascending=False).head(6)
            # Colour bar chart using HTML for a nicer look
            for style_name, prob in prob_series.items():
                bar_w = int(prob * 100)
                col_hex = STYLE_COLORS.get(style_name, "#888")
                is_detected = style_name == res["detected"]
                label_weight = "600" if is_detected else "400"
                st.markdown(f"""
                <div style="margin-bottom:0.6rem">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem">
                        <span style="font-size:0.82rem;font-weight:{label_weight};color:#1A1A1A">
                            {STYLE_EMOJI.get(style_name,'')} {style_name}{'  ✓' if is_detected else ''}
                        </span>
                        <span style="font-size:0.78rem;color:#9B9B9B">{prob:.0%}</span>
                    </div>
                    <div style="background:#F0F0EE;border-radius:4px;height:6px;overflow:hidden">
                        <div style="background:{col_hex};width:{bar_w}%;height:100%;border-radius:4px;transition:width 0.5s"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.markdown('<p class="section-title">Colour palette</p>', unsafe_allow_html=True)
            swatches = "".join(
                f"<span class='swatch' style='background:rgb{c}' title='{s:.0%}'></span>"
                for c, s in res["palette"]
            )
            st.markdown(swatches, unsafe_allow_html=True)

            st.markdown('<p class="section-title" style="margin-top:1.5rem">What we found</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="analysis-row">
                <div class="analysis-label">In your room</div>
                <div class="analysis-value">{", ".join(sorted(res["have"])) or "Nothing recognised"}</div>
            </div>
            <div class="analysis-row">
                <div class="analysis-label">Suggested additions</div>
                <div class="analysis-value" style="color:#C0552A">{", ".join(res["missing"]) or "Room is well furnished!"}</div>
            </div>
            """, unsafe_allow_html=True)
