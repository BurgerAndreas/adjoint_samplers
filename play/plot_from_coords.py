import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

import py3Dmol
from pyppeteer import launch
from rdkit import Chem
from rdkit.Geometry import Point3D


def build_ethanol() -> Chem.Mol:
    """Construct ethanol with explicit 3D coordinates."""
    symbols = [
        "C",
        "C",
        "O",  # heavy atoms
        "H",
        "H",
        "H",  # hydrogens on first carbon
        "H",
        "H",  # hydrogens on second carbon
        "H",  # hydroxyl hydrogen
    ]
    coords = [
        (0.0, 0.0, 0.0),  # C1
        (1.53, 0.0, 0.0),  # C2
        (2.80, 0.20, 0.0),  # O
        (-0.54, 0.94, 0.0),  # H (C1)
        (-0.54, -0.47, 0.82),  # H (C1)
        (-0.54, -0.47, -0.82),  # H (C1)
        (1.93, 0.93, 0.0),  # H (C2)
        (1.93, -0.46, 0.82),  # H (C2)
        (3.24, -0.64, 0.0),  # H (O)
    ]
    bonds = [
        (0, 1),
        (1, 2),  # backbone
        (0, 3),
        (0, 4),
        (0, 5),  # C1 hydrogens
        (1, 6),
        (1, 7),  # C2 hydrogens
        (2, 8),  # hydroxyl hydrogen
    ]

    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(sym))
    for begin, end in bonds:
        rw.AddBond(begin, end, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    conf = Chem.Conformer(len(symbols))
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    return mol


def mol_to_view(mol: Chem.Mol) -> py3Dmol.view:
    """Create a styled py3Dmol view from an RDKit molecule."""
    view = py3Dmol.view(width=800, height=600)
    view.addModel(Chem.MolToMolBlock(mol), "sdf")
    view.setBackgroundColor("white")
    view.setStyle({"stick": {"radius": 0.2, "colorscheme": "Jmol"}})
    view.addPropertyLabels(
        "symbol",
        {"not": {"elem": "C"}},
        {
            "fontColor": "black",
            "font": "sans-serif",
            "fontSize": 16,
            "showBackground": False,
            "alignment": "center",
        },
    )
    view.zoomTo()
    return view


async def main():
    mol = build_ethanol()
    view = mol_to_view(mol)

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
    await page.screenshot(path="plots/ethanol_coords.png")
    await browser.close()
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
