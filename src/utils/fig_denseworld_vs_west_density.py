"""Bold the TEXT of paper Figure 5 (``fig:denseworld_vs_west_density``).

The figure is a professor-provided raster PDF
(``figures/denseworld_vs_west_density.png.pdf``); its low-weight text is hard to
read at column width. Text size cannot be changed on a flat bitmap without redrawing
the chart (which would move the bars), so instead we thicken only the dark, un-colored
pixels (black text + axis lines): the colored bars are high-saturation and are excluded
by construction, so the bar heights are byte-for-byte preserved. Output resolution and
page geometry match the source, so the LaTeX layout does not shift.

Requires Ghostscript (``gs``) on PATH for cropbox-accurate rasterization.

USAGE:
    python -u src/utils/fig_denseworld_vs_west_density.py \\
        --source overleaf/2026___FactorJEPA_AAAI/figures/denseworld_vs_west_density.png.pdf \\
        --output overleaf/2026___FactorJEPA_AAAI/figures/denseworld_vs_west_density_bold.pdf
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RENDER_DPI = 600      # rasterization resolution for the source PDF
DARK_MAX = 120        # luminance below this = candidate text/axis pixel
SAT_MAX = 50          # (max-min) channel spread below this = un-colored (not a bar)
DILATE = 5            # MaxFilter kernel (odd) -> how much strokes thicken


def _render_cropbox(source: Path, dpi: int) -> Image.Image:
    """Rasterize the source PDF honoring its crop box (the region LaTeX shows)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "page.png"
        subprocess.run(
            ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dUseCropBox", "-sDEVICE=png16m",
             f"-r{dpi}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
             "-o", str(out), str(source)],
            check=True,
        )
        return Image.open(out).convert("RGB").copy()


def embolden_text(source: Path, out_pdf: Path) -> None:
    """Thicken black text/axis pixels only; colored bars are left identical."""
    a = np.asarray(_render_cropbox(source, RENDER_DPI)).astype(int)
    gray = a.mean(2)
    sat = a.max(2) - a.min(2)
    dark_text = (gray < DARK_MAX) & (sat < SAT_MAX)          # black/gray text + axes
    mask = Image.fromarray((dark_text * 255).astype("uint8")).filter(ImageFilter.MaxFilter(DILATE))
    out = a.copy()
    out[np.asarray(mask) > 0] = [0, 0, 0]                    # solid black -> bold
    Image.fromarray(out.astype("uint8")).save(out_pdf, "PDF", resolution=float(RENDER_DPI))
    print(f"Wrote: {out_pdf}  (bars unchanged; text/axes thickened)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bold the text of paper Figure 5 (bars untouched).")
    p.add_argument("--source", type=Path, required=True, help="professor's source .png.pdf")
    p.add_argument("--output", type=Path, required=True, help="output bold .pdf path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.exists():
        sys.exit(f"FATAL: --source not found: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embolden_text(args.source, args.output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\nFATAL: {Path(__file__).name} crashed - see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
