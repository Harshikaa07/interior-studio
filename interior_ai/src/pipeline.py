from . import config
from .palette import dominant_colors
from .segment import detected_categories


def analyze_room(img, m, target=None):
    """Everything we learn about a room photo + the redesigned image."""
    room_emb = m["emb"].images([img])[0]
    probs = m["style"].predict(room_emb)
    detected = max(probs, key=probs.get)
    objs = m["seg"].objects(img)
    have = detected_categories(objs)
    target = detected if target in (None, "Same as detected") else target
    return dict(original=img, room_emb=room_emb, probs=probs, detected=detected,
                palette=dominant_colors(img), objects=objs, have=have,
                missing=[c for c in config.ESSENTIALS if c not in have],
                target=target, redesign=m["redesign"].run(img, target))
