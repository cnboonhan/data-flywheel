#!/usr/bin/env python3
"""Tile every bufo (assets/bufo submodule) into one image: the Keycloak login page's background.

    uv run --with pillow tools/bufo_mosaic.py [out.png] [cell px]
    # default out: $STATE_DIR/keycloak/login-background/background.png (outside git; the bufos carry no licence)

Animated GIFs contribute their first frame. Cells are square; images keep their aspect ratio inside them. The result is
flattened onto the page's dark background colour (#1f2328) with a 256-colour palette to keep it small.
"""

import math
import os
import sys
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[1]
src = root / "assets" / "bufo" / "all-the-bufo"
state = os.environ.get("STATE_DIR", "/tier1/htx_boonhan/services")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(state) / "keycloak" / "login-background" / "background.png"
cell = int(sys.argv[2]) if len(sys.argv) > 2 else 48

files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".gif", ".jpg", ".jpeg"))
if not files:
    sys.exit(f"no images in {src}; run: git submodule update --init assets/bufo")
cols = math.ceil(math.sqrt(len(files)))
rows = math.ceil(len(files) / cols)
mosaic = Image.new("RGBA", (cols * cell, rows * cell), (0, 0, 0, 0))
for i, p in enumerate(files):
    try:
        im = Image.open(p)
        im.seek(0)
        im = im.convert("RGBA")
    except Exception as e:  # a broken file shouldn't stop the mosaic
        print(f"skip {p.name}: {e}", file=sys.stderr)
        continue
    im.thumbnail((cell, cell), Image.LANCZOS)
    x, y = (i % cols) * cell + (cell - im.width) // 2, (i // cols) * cell + (cell - im.height) // 2
    mosaic.alpha_composite(im, (x, y))
# Flatten onto the login page's dark background and use a 256-colour palette: a few MB less for every login.
flat = Image.new("RGB", mosaic.size, (0x1f, 0x23, 0x28))
flat.paste(mosaic, mask=mosaic.getchannel("A"))
out.parent.mkdir(parents=True, exist_ok=True)
flat.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
print(f"{out}: {len(files)} bufos, {cols}x{rows} cells of {cell}px, {mosaic.width}x{mosaic.height}, {out.stat().st_size // 1024} KB")
