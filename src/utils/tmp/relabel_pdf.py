"""Label-only surgery on a matplotlib vector PDF (works on a COPY).

1) Redact the edited label lines -> the OLD glyphs are removed from the text layer
   (so a text audit finds no ViT-G/POC left), but this can collaterally damage a
   sibling text object (e.g. a 2-line suptitle sharing one object).
2) Detect any UNEDITED line whose text vanished and restore it.
3) White-box + redraw (font-matched DejaVu) the edited lines with new text and the
   restored lines with their original text. Horizontal via TextWriter at exact
   baseline+center; vertical (rotated y-axis) via insert_textbox(rotate=90).
Plot geometry (bars/axes/values) is never touched — only margin label lines match.
"""
import re
import sys
import fitz
from matplotlib import font_manager

SRC, DST = sys.argv[1], sys.argv[2]
REG_PATH = font_manager.findfont("DejaVu Sans")
BOLD_PATH = font_manager.findfont("DejaVu Sans:bold")
REG_F = fitz.Font(fontfile=REG_PATH)
BOLD_F = fitz.Font(fontfile=BOLD_PATH)
REG_BUF = open(REG_PATH, "rb").read()
BOLD_BUF = open(BOLD_PATH, "rb").read()

MID = "·"
_G2B = re.compile(r"ViT-G.{0,4}2B")
_g1B = re.compile(r"ViT-g.{0,4}1B")


def newtext(joined):
    j = joined.strip()
    if j == "ntotal = 10k":
        return "Stratified 10k"
    if j == "ntotal = 116k":
        return "Full 115k"
    if "V-JEPA 2.1" in j and "ViT-G" in j and "48 blocks" in j:
        return f"vJEPA2.1 (2B)   {MID}   1664-dim, 48 blocks   (CHAMPION)"
    if "V-JEPA 2.1" in j and "ViT-g" in j and "40 blocks" in j:
        return f"vJEPA2.1 (1B)   {MID}   1408-dim, 40 blocks"
    if len(j) < 22 and ("ViT-G" in j or "ViT-g" in j):
        return "vJEPA2.1 (2B)" if "ViT-G" in j else "vJEPA2.1 (1B)"
    out = joined
    out = out.replace("FULL ViT-G 2B", "Full-scale 2B")
    out = out.replace("FULL 1B", "115k")
    out = _G2B.sub("vJEPA2.1 (2B)", out)
    out = _g1B.sub("vJEPA2.1 (1B)", out)
    out = out.replace("POC 10k", "Stratified 10k")
    out = out.replace("POC", "Stratified 10k")
    out = out.replace("116k", "115k")
    return out


def _place(page, text, L):
    r = L["r"]
    ffont = BOLD_F if L["bold"] else REG_F
    fname = "dvb" if L["bold"] else "dvr"
    fs = L["size"]
    if not L["vertical"]:
        w = ffont.text_length(text, fontsize=fs)
        maxw = r.width * 1.5
        if w > maxw:
            fs *= maxw / w
            w = maxw
        cx = (r.x0 + r.x1) / 2
        tw = fitz.TextWriter(page.rect)
        tw.append((cx - w / 2, L["origin"][1]), text, font=ffont, fontsize=fs)
        tw.write_text(page, color=L["rgb"])
    else:
        h = ffont.text_length(text, fontsize=fs)
        maxh = r.height * 1.5
        if h > maxh:
            fs *= maxh / h
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        box = fitz.Rect(cx - fs * 1.2, cy - r.height, cx + fs * 1.2, cy + r.height)
        page.insert_textbox(box, text, fontname=fname, fontsize=fs, color=L["rgb"],
                            align=fitz.TEXT_ALIGN_CENTER, rotate=90)


def main():
    doc = fitz.open(SRC)
    total = 0
    for page in doc:
        page.insert_font(fontname="dvr", fontbuffer=REG_BUF)
        page.insert_font(fontname="dvb", fontbuffer=BOLD_BUF)
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                if not spans:
                    continue
                joined = "".join(s["text"] for s in spans)
                x0 = min(s["bbox"][0] for s in spans); y0 = min(s["bbox"][1] for s in spans)
                x1 = max(s["bbox"][2] for s in spans); y1 = max(s["bbox"][3] for s in spans)
                m = max(spans, key=lambda s: s["size"])
                col = m["color"]
                d = line.get("dir", (1.0, 0.0))
                lines.append({
                    "joined": joined, "nt": newtext(joined), "r": fitz.Rect(x0, y0, x1, y1),
                    "size": m["size"], "rgb": ((col >> 16 & 255) / 255, (col >> 8 & 255) / 255, (col & 255) / 255),
                    "bold": bool(m["flags"] & 16) or "Bold" in m.get("font", ""),
                    "vertical": abs(d[1]) > 0.7, "origin": m["origin"],
                })
        edited = [L for L in lines if L["nt"] != L["joined"]]
        for L in edited:
            page.add_redact_annot(L["r"], fill=(1, 1, 1))
        if edited:
            page.apply_redactions()
        remaining = page.get_text()
        for L in lines:
            if L["nt"] != L["joined"]:
                # only cover with white when redaction did NOT remove the old glyphs
                # (matplotlib XObject text) — otherwise the box needlessly clips the
                # neighbouring line above.
                if L["joined"].strip() in remaining:
                    page.draw_rect(L["r"] + (-0.4, -0.4, 0.4, 0.4), color=None, fill=(1, 1, 1))
                _place(page, L["nt"], L)
                total += 1
            elif L["joined"].strip() and L["joined"].strip() not in remaining:
                # a sibling line the redaction collaterally removed -> clear + restore it
                page.draw_rect(L["r"] + (-0.4, -0.4, 0.4, 0.4), color=None, fill=(1, 1, 1))
                _place(page, L["joined"], L)
    doc.save(DST, garbage=4, deflate=True)
    print(f"wrote {DST} ({total} lines relabelled)")


main()
