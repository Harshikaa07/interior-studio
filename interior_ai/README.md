# 🏠 Interior Studio

> AI-powered room visualizer and furniture recommender — upload a photo, pick an aesthetic, get a redesign and curated shop links.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What it does

1. **Analyze** — SegFormer detects objects in your room; CLIP extracts a style embedding; K-means finds the dominant colour palette.
2. **Redesign** — Stable Diffusion + ControlNet (canny edges) repaints the room in your chosen aesthetic, preserving walls, windows, and layout. CPU fallback applies a colour-grade preview.
3. **Recommend** — A hybrid CLIP-based retrieval scores products by room similarity (30%), style match (30%), colour harmony (20%) and your taste (20%). Each recommendation links directly to a shop.

---

## Datasets used

| Dataset | Role | How to get |
|---|---|---|
| **Wayfair WANDS** | Product catalogue (names, categories) — auto-fetched | `python datasets.py fetch` |
| **Amazon Berkeley Objects (ABO)** | Product catalogue (photos + names + colours) — auto-fetched | `python datasets.py fetch` |
| **Pinterest Shop The Look** | Evaluation: scene→product retrieval | `python datasets.py fetch` |
| **Kaggle Interior Design Styles** | Labelled data for the style classifier | See [setup](#dataset-setup) |
| **Houzz Interior Design (Kaggle)** | Additional labelled style images (15 styles) | See [setup](#dataset-setup) |
| **IKEA furniture (Kaggle)** | Product catalogue with prices | See [setup](#dataset-setup) |
| **Places365** (indoor split) | Room-photo pool for pseudo-labelling | Auto (torchvision) |

---

## Pipeline architecture

```
Room photo
    │
    ├── CLIP image encoder ──────────────────────────────────────────► room embedding
    │       │                                                               │
    │   style head (logistic regression)                                    │
    │       │                                                               │
    │   detected style                                                      │
    │                                                                       │
    ├── SegFormer (ADE20K) ──► object labels ──► detected categories        │
    │                                                                       │
    └── K-means (5 clusters) ──► dominant colours                           │
                                                                            ▼
                                                                  Hybrid Recommender
                                                                 (room + style + colour
                                                                  + taste → ranked list)
                                                                            │
Target style ──► Stable Diffusion + ControlNet ──► Redesigned room         │
                                                                            ▼
                                                               Product cards + shop links
```

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/Harshikaa07/interior-studio.git
cd interior-studio

# 2. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the auto-fetchable datasets
python datasets.py fetch

# 5. Build the product index
python build_index.py

# 6. Run the app
streamlit run app.py
```

> **No GPU?**  The app still works — redesign uses a quick colour-grade preview instead of Stable Diffusion. For the full AI redesign, run on Google Colab with a T4 GPU (see below).

---

## Dataset setup

### Auto-fetched (no action needed)
```bash
python datasets.py fetch   # downloads ABO listings, WANDS, Shop The Look, Places365
```

### Kaggle datasets (optional but recommended)

1. Go to kaggle.com → profile → **Settings** → **API** → **Create New Token** → download `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json`  (Windows: `C:\Users\<you>\.kaggle\kaggle.json`)
3. Run:

```bash
pip install kaggle

# Interior Design Styles — labelled style images
python datasets.py kaggle natecat/interior-design-styles styles

# Houzz dataset — 15 style categories
python datasets.py kaggle meherunnesashoshi/houzz-interior-design-dataset houzz

# IKEA furniture — names, categories and prices
python datasets.py kaggle ahmedkallam/ikea-sa-furniture-web-scraping ikea
```

> Find the exact slug (`owner/dataset`) from the dataset's URL on kaggle.com: `kaggle.com/datasets/OWNER/NAME`.

After adding any dataset:
```bash
python build_index.py    # re-build the product index
python train_style.py    # re-train the style classifier (needs data/styles/ or data/houzz/)
```

---

## Run on Google Colab (free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com) → New notebook → Runtime → Change runtime type → **T4 GPU** → Save.
2. Upload the project zip into Colab's file browser.
3. Run each cell:

```bash
!unzip -q interior-studio.zip
%cd interior-studio
!pip install -q -r requirements.txt
!python datasets.py fetch
!python build_index.py
```

4. *(Optional)* Add Kaggle datasets as above.

5. Launch the app:

```bash
!pip install -q localtunnel
!streamlit run app.py &> /content/log.txt &
!curl ipv4.icanhazip.com
!npx localtunnel --port 8501
```

Open the `https://...loca.lt` URL and paste the IP from `curl` as the password on first visit.

---

## Train the style classifier

```bash
python train_style.py    # prints: zero-shot CLIP accuracy vs trained-head accuracy
```

Training uses:
- **Labelled images** from `data/styles/` and `data/houzz/` (folder names map to style names automatically).
- **Pseudo-labelled images** from Places365 / LSUN / ADE20K (confident CLIP predictions above 0.5 confidence are kept at half weight).

---

## Evaluate retrieval quality

```bash
python evaluate.py stl 1000   # recall@1/5/10 on Pinterest Shop The Look
```

---

## Run the tests

```bash
python tests_core.py   # no model downloads needed — uses fake CLIP/segmenter
```

---

## Notes & limitations

- **Prices** are shown in Indian Rupees (Rs.). Only IKEA rows carry actual prices; ABO and WANDS have no price, so the budget slider only filters IKEA items.
- **Shop links** are generated search URLs (Wayfair, Amazon.in, IKEA) — not guaranteed affiliate links.
- **Redesign accuracy**: the redesign is generative, not measured-to-scale. Structural features (walls, windows) are preserved via ControlNet canny edges; furniture placement is approximated.
- **Shop The Look images** are 2019 Pinterest links — some are dead. Use a larger N for the evaluation (`python evaluate.py stl 2000`).
- **Gated datasets** (3D-FRONT, Structured3D, Matterport3D) require access approval from their respective maintainers.

---

## Project structure

```
interior-studio/
├── app.py               # Streamlit UI
├── datasets.py          # Dataset download & status
├── build_index.py       # Embed all products into CLIP space
├── train_style.py       # Train the style classifier
├── evaluate.py          # Retrieval evaluation (Shop The Look)
├── tests_core.py        # Logic tests (no downloads)
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py        # Styles, categories, currency, shop-link generators
    ├── embed.py         # CLIP embedder (images + text)
    ├── style.py         # Style classifier (zero-shot + trained head)
    ├── segment.py       # SegFormer object detector
    ├── palette.py       # K-means colour palette + Lab colour harmony
    ├── generate.py      # SD + ControlNet redesigner (CPU colour-grade fallback)
    ├── recommend.py     # Hybrid CLIP retrieval recommender
    ├── catalog.py       # Dataset loaders (ABO, IKEA, WANDS) + shop links
    ├── rooms.py         # Room-image pool loaders for training
    └── pipeline.py      # End-to-end room analysis
```
