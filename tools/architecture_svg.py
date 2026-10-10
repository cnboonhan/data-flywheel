#!/usr/bin/env python3
"""Write architecture.svg from architecture.html (the source), so the README can show the diagram inline.

    python3 tools/architecture_svg.py [assets/architecture.html [assets/architecture.svg]]

GitHub renders READMEs without iframes, scripts or page styles, so the HTML can't be embedded; the SVG inside it can,
once its styles travel with it. Light and dark follow the viewer's colour scheme. Links aren't clickable in a README
image; the README links to the HTML for that.
"""

import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "assets" / "architecture.html"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "assets" / "architecture.svg"
html = src.read_text()
css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
svg = re.search(r"<svg\b.*?</svg>", html, re.S).group(0)

# Keep the colour variables and the rules for SVG classes; drop page layout (body, main, h1, figure, legend).
keep = [r for r in re.findall(r"[^{}]+\{[^{}]*\}", re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css))
        if r.strip().startswith((":root", "."))]
dark = re.search(r"@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme=\"light\"\]\) (\{[^}]*\})", css).group(1)
style = "\n".join(r.strip() for r in keep if "data-theme" not in r)
style += "\n@media (prefers-color-scheme: dark) { :root " + dark + " }"
style += "\nsvg { color: var(--fg); font: 15px/1.5 system-ui, -apple-system, \"Segoe UI\", sans-serif; }"
style += "\n.bg { fill: var(--bg); }"

view = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
w, h = view.groups()
svg = svg.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ', 1)
svg = re.sub(r"(<svg\b[^>]*>)", rf'\1\n<style>\n{style}\n</style>\n<rect class="bg" width="{w}" height="{h}"/>', svg, count=1)
dst.write_text("<!-- Generated from architecture.html by tools/architecture_svg.py; edit the HTML. -->\n" + svg + "\n")
print(f"{dst.name} ({w}x{h})")
