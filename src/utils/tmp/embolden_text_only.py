"""Bold the TEXT/axes of the professor's rasterized figure, leaving the colored
bars as the identical pixels (bar heights cannot change - we never touch them)."""
import sys
import numpy as np
from PIL import Image, ImageFilter

src_png, out_pdf, dpi = sys.argv[1], sys.argv[2], float(sys.argv[3])

im = Image.open(src_png).convert("RGB")
a = np.asarray(im).astype(int)
gray = a.mean(2)
sat = a.max(2) - a.min(2)
# black/gray text + axis lines only (low luminance AND near-gray);
# colored bars are high-saturation -> excluded; light gridlines are bright -> excluded
dark_text = (gray < 120) & (sat < 50)
mask = Image.fromarray((dark_text * 255).astype("uint8")).filter(ImageFilter.MaxFilter(5))
m = np.asarray(mask) > 0
out = a.copy()
out[m] = [0, 0, 0]                      # solid black -> bold, crisp; bars never in mask
Image.fromarray(out.astype("uint8")).save(out_pdf, "PDF", resolution=dpi)
print(f"wrote {out_pdf}  size={im.size}")
