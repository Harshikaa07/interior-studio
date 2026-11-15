# Interior Studio: AI interior visualizer + recommender

Upload a room photo → the app detects its style, colours and missing furniture, redesigns the room in a
chosen style, and recommends products that fit (prices in ₹). 👍/👎 personalises the list.

## EXACT STEPS: Google Colab (recommended, free GPU)

1. Open colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU** → Save.
2. Click the folder icon on the left, drag `interior_ai.zip` into it.
3. Run each of these in its own cell:
```
!unzip -q interior_ai.zip
%cd interior_ai
!pip install -q -r requirements.txt
!python datasets.py fetch
!python datasets.py status
```
4. **Kaggle datasets** (style images + IKEA). On kaggle.com: profile picture → Settings → API → *Create New Token*
   (downloads `kaggle.json`). Upload it to Colab, then:
```
!pip install -q kaggle
!mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!python datasets.py kaggle OWNER/INTERIOR-STYLES-DATASET styles
!python datasets.py kaggle OWNER/IKEA-DATASET ikea
```
   Find the `OWNER/NAME` by searching Kaggle for "interior design styles" and "IKEA furniture dataset" (the slug is
   in the dataset's URL: kaggle.com/datasets/OWNER/NAME). Style images must end up in folders whose name contains a
   style (modern, scandinavian, ...). If a folder has another name, create `data/styles/style_map.json`, e.g.
   `{"dark_wood": "Traditional"}`. Run `python train_style.py` and it prints any folders it couldn't match.
5. Optional extra scene datasets (LSUN, ADE20K, SUN RGB-D; Structured3D/3D-FRONT/Matterport3D need access approval):
   put their images into `data/<name>/` (names listed by `python datasets.py status`). Skip any you don't have.
6. Build and train:
```
!python build_index.py            # embeds the product catalogue (ABO + IKEA + WANDS)
!python train_style.py            # trains the style classifier, prints accuracy vs zero-shot (needs data/styles)
!python evaluate.py stl 1000      # retrieval score on Pinterest Shop The Look; copy these numbers into your report
```
7. Launch the app:
```
!npm install -g localtunnel -q
!streamlit run app.py &>/content/log.txt &
!curl ipv4.icanhazip.com
!npx localtunnel --port 8501
```
   Open the `https://….loca.lt` link; when it asks for a password, paste the IP printed by `curl`.
   The first load downloads the AI models (a few minutes).

## EXACT STEPS: your own laptop (works, but redesign is a quick colour-grade preview without an NVIDIA GPU)
```
python -m venv venv
venv\Scripts\activate            (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python datasets.py fetch
python build_index.py
streamlit run app.py
```
Works with zero extra datasets (WANDS is fetched automatically; a tiny demo catalogue is used if nothing is found).
After adding any dataset, run `python build_index.py` again.

## Where each dataset is used
| Dataset | Role |
|---|---|
| Kaggle interior styles | labelled data for the style classifier |
| Places365, LSUN, SUN RGB-D, ADE20K, (3D-FRONT, Structured3D, Matterport3D) | room-photo pool, pseudo-labelled by CLIP to enlarge style training; ADE20K also trained the SegFormer |
| Amazon Berkeley Objects, IKEA, Wayfair WANDS | product catalogue (photo-embedded when images exist, text-embedded otherwise) |
| Pinterest Shop The Look (home) | evaluation of room → product retrieval |

## Pipeline
1. **Understand**: SegFormer (objects), K-means (palette), CLIP + logistic-regression head (style)
2. **Redesign**: Stable Diffusion + ControlNet (canny) keeps the layout; CPU fallback = style colour-grade
3. **Recommend**: score = 0.30 room similarity + 0.30 style match + 0.20 colour harmony (Lab space) + 0.20 taste
   (liked/disliked items), + gap bonus for categories the room lacks, budget filter, ≤2 items per category

## Limits (say these in your report)
- Prices: only IKEA rows carry prices (converted to ₹ at the rates in `src/config.py`, approximate Nov 2026:
  1 SAR ≈ ₹25.5, 1 USD ≈ ₹95). ABO and WANDS have no price, so the budget slider can't filter them.
- The common IKEA Kaggle CSV has no currency column; it's assumed to be SAR (`IKEA_DEFAULT_CURRENCY`). If your CSV has a
  `currency` column it is used instead.
- Shop The Look images are 2019 Pinterest links and some are dead; `evaluate.py` reports how many downloaded.
- Gated datasets need access approval; the code simply uses whatever is in `data/<name>/`.
- The redesign is a generative approximation, not a measured-to-scale design.
- `python tests_core.py` runs the logic tests (no downloads needed).
