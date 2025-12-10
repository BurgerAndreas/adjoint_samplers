import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

import py3Dmol
from pyppeteer import launch

# Example: Caffeine
cid = "2519"
# Width/height set for a consistent PNG snapshot on headless servers.
view = py3Dmol.view(query=f"cid:{cid}", width=800, height=600)

# 1. Clean white background for a flat comic vibe
view.setBackgroundColor("white")

# 2. Pastel element palette to look more illustrative than realistic
PASTEL_BY_ELEM = {
    "C": "#9bb3c7",  # muted blue-gray
    "H": "#d9d9d9",  # light gray
    "N": "#9bc1bc",  # pastel teal
    "O": "#f28c8c",  # soft coral
    "S": "#f6d186",  # pale amber
}

# 3. Sticks kept slightly thicker for bold outlines; gentle opacity softens 3D
view.setStyle(
    {
        "stick": {
            "radius": 0.22,
            "opacity": 0.9,
            "colorscheme": {"prop": "elem", "map": PASTEL_BY_ELEM},
        }
    }
)

# 4. Labels on heteroatoms; no background to keep it airy
view.addPropertyLabels(
    "symbol",
    {"not": {"elem": "C"}},
    {
        "fontColor": "#4a4a4a",
        "font": "sans-serif",
        "fontSize": 16,
        "showBackground": False,
        "alignment": "center",
    },
)

view.zoomTo()


async def main():
    html = view.write_html()
    with NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(html.encode("utf-8"))

    browser = await launch(
        headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
    page = await browser.newPage()
    await page.setViewport({"width": 800, "height": 600})
    await page.goto(tmp_path.as_uri(), waitUntil="networkidle0")
    await page.waitFor(1000)
    await page.screenshot(path="plots/caffeine.png")
    await browser.close()
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
