import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX = ROOT / "index"
DATA.mkdir(exist_ok=True)
INDEX.mkdir(exist_ok=True)

CLIP_MODEL = "openai/clip-vit-base-patch32"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"   # trained on ADE20K
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_MODEL = "lllyasviel/sd-controlnet-canny"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

STYLES = {
    "Modern":        "clean lines, neutral palette, glass and metal accents, uncluttered",
    "Scandinavian":  "light wood, white walls, cozy textiles, simple functional furniture",
    "Minimalist":    "very few objects, monochrome palette, open space, hidden storage",
    "Industrial":    "exposed brick, black metal, concrete, leather, raw wood",
    "Bohemian":      "layered rugs, rattan, plants, colorful patterned textiles",
    "Rustic":        "reclaimed wood, stone, warm earth tones, handmade pieces",
    "Mid-century":   "teak furniture, tapered legs, organic shapes, retro accents",
    "Traditional":   "classic furniture, rich wood, symmetry, elegant fabrics",
    "Coastal":       "white and blue palette, light linen, natural fibers, airy",
    "Contemporary":  "bold geometry, mixed materials, statement lighting",
    "Art Deco":      "geometric patterns, gold accents, velvet, lacquered surfaces",
    "Japandi":       "wabi-sabi, natural materials, calm neutrals, functional minimalism",
}

ESSENTIALS = ["sofa", "chair", "table", "lighting", "rug", "curtain", "storage", "decor"]

# order matters: first match wins ("table lamp" -> lighting)
CATEGORY_KEYWORDS = {
    "lighting":  ["lamp", "light", "chandelier", "pendant", "sconce", "lantern"],
    "rug":       ["rug", "carpet"],
    "curtain":   ["curtain", "drape", "blind"],
    "sofa":      ["sofa", "couch", "loveseat", "sectional"],
    "bed":       ["bed", "mattress"],
    "chair":     ["chair", "stool", "armchair", "recliner", "ottoman", "bench"],
    "table":     ["table", "desk"],
    "storage":   ["cabinet", "shelf", "shelves", "shelving", "wardrobe", "dresser",
                  "bookcase", "drawer", "drawers", "storage", "sideboard"],
    "decor":     ["mirror", "vase", "art", "painting", "frame", "cushion", "pillow",
                  "plant", "clock", "sculpture"],
}
_CAT_RE = {c: re.compile(r"\b(?:" + "|".join(k) + r")s?\b") for c, k in CATEGORY_KEYWORDS.items()}


def categorize(text):
    t = str(text).lower()
    for cat, rx in _CAT_RE.items():
        if rx.search(t):
            return cat
    return None


COLOR_WORDS = {
    "white":   (240, 240, 236), "black": (30, 30, 30),    "grey":    (128, 128, 128),
    "gray":    (128, 128, 128), "beige": (222, 205, 175),  "brown":   (110, 75, 50),
    "walnut":  (95, 65, 45),    "oak":   (190, 150, 100),  "natural": (210, 185, 150),
    "blue":    (60, 100, 170),  "navy":  (30, 45, 90),     "green":   (80, 130, 90),
    "red":     (170, 50, 50),   "yellow":(225, 190, 60),   "pink":    (220, 150, 170),
    "orange":  (220, 130, 50),  "gold":  (200, 165, 70),   "silver":  (190, 190, 195),
    "cream":   (245, 235, 210), "ivory": (250, 245, 230),  "teal":    (40, 125, 125),
    "purple":  (120, 70, 150),  "tan":   (200, 170, 130),
}

# All prices are shown in Indian Rupees.
# Approximate mid-Nov 2026 rates (update if needed).
CURRENCY_TO_INR = {"INR": 1.0, "SAR": 25.5, "USD": 95.0, "EUR": 106.0, "GBP": 127.0}
IKEA_DEFAULT_CURRENCY = "SAR"   # common IKEA Kaggle CSV has no currency column; prices are Saudi riyals


def rupees(x):
    """Format a number as Indian Rupees (ASCII-safe)."""
    return f"Rs.{x:,.0f}"


def style_prompt(style):
    return f"a {style.lower()} style interior, {STYLES[style]}, professional interior photography"


# ---------------------------------------------------------------------------
# Shopping link generators
# ---------------------------------------------------------------------------

def _slug(text):
    return re.sub(r"\s+", "+", str(text).strip()[:80])


def wayfair_link(name, category=""):
    q = _slug(f"{name} {category}")
    return f"https://www.wayfair.com/keyword.php?keyword={q}"


def amazon_link(name, category=""):
    q = _slug(f"{name} {category} furniture")
    return f"https://www.amazon.in/s?k={q}"


def ikea_link(name="", category=""):
    q = _slug(f"{name} {category}")
    return f"https://www.ikea.com/in/en/search/?q={q}"


def google_shopping_link(name, category=""):
    q = _slug(f"{name} {category} buy online India")
    return f"https://www.google.com/search?tbm=shop&q={q}"
