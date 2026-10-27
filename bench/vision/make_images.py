"""The image set of VISION §20.8 (bench/vision/README.md).

  make_images.py rendered OUT_DIR     the screenshots, the code screenshot, the chart and the icon, drawn here (committed
                                      under bench/vision/images/; DejaVu fonts, Pillow)
  make_images.py derived OUT_DIR      the photos from the Ubuntu/DGX OS wallpapers in /usr/share/backgrounds, checked by
                                      SHA-256, as clients send them: the originals, and a copy resized to 1600 px wide
                                      and re-encoded as Codex does (JPEG at quality 85)
"""
import hashlib
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONTS = "/usr/share/fonts/truetype/dejavu/"


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


def terminal(path):
    img = Image.new("RGB", (1280, 720), (30, 30, 36))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 32], fill=(52, 52, 60))
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([14 + i * 22, 9, 28 + i * 22, 23], fill=c)
    d.text((560, 7), "dev@node-b: ~/proj", font=font("DejaVuSans.ttf", 14), fill=(200, 200, 205))
    mono = font("DejaVuSansMono.ttf", 17)
    lines = [
        ((120, 220, 120), "$ cargo build --release"),
        ((220, 220, 220), "   Compiling parser v0.4.2 (/work/proj/parser)"),
        ((240, 90, 90), "error[E0502]: cannot borrow `tokens` as mutable because it is also borrowed as immutable"),
        ((120, 160, 240), "  --> parser/src/lexer.rs:142:17"),
        ((120, 160, 240), "   |"),
        ((220, 220, 220), "139 |         let first = tokens.first().unwrap();"),
        ((240, 200, 90), "    |                     ------ immutable borrow occurs here"),
        ((220, 220, 220), "...  "),
        ((220, 220, 220), "142 |                 tokens.push(Token::Eof);"),
        ((240, 90, 90), "    |                 ^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here"),
        ((220, 220, 220), "143 |         println!(\"{:?}\", first);"),
        ((240, 200, 90), "    |                          ----- immutable borrow later used here"),
        ((220, 220, 220), ""),
        ((240, 90, 90), "error: could not compile `parser` (lib) due to 1 previous error"),
        ((120, 220, 120), "$ "),
    ]
    y = 48
    for color, text in lines:
        d.text((16, y), text, font=mono, fill=color)
        y += 26
    img.save(path, optimize=False)


def code(path):
    img = Image.new("RGB", (1600, 900), (40, 44, 52))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 260, 900], fill=(33, 37, 43))
    ui = font("DejaVuSans.ttf", 15)
    for i, name in enumerate(["src/", "  engine.cpp", "  engine.hpp", "  kernels.cu", "  vision.cu", "tests/", "CMakeLists.txt"]):
        d.text((14, 50 + i * 26), name, font=ui, fill=(171, 178, 191) if i != 4 else (255, 255, 255))
    d.rectangle([260, 0, 1600, 36], fill=(33, 37, 43))
    d.text((280, 9), "vision.cu", font=ui, fill=(255, 255, 255))
    mono = font("DejaVuSansMono.ttf", 18)
    src = [
        ("kw", "template <typename T>"),
        ("tx", "T clamp_add(const std::vector<T>& xs, T lo, T hi) {"),
        ("tx", "  T acc = 0;"),
        ("kw", "  for (size_t i = 0; i <= xs.size(); ++i) {"),
        ("tx", "    acc += xs[i];"),
        ("tx", "    if (acc > hi) acc = hi;"),
        ("tx", "    if (acc < lo) acc = lo;"),
        ("tx", "  }"),
        ("kw", "  return acc;"),
        ("tx", "}"),
        ("tx", ""),
        ("cm", "// Returns the mean of the window ending at i (inclusive)."),
        ("tx", "double window_mean(const std::vector<double>& v, int i, int w) {"),
        ("tx", "  double s = 0;"),
        ("kw", "  for (int k = i - w; k < i; ++k) s += v[k];"),
        ("kw", "  return s / w;"),
        ("tx", "}"),
    ]
    colors = {"kw": (198, 120, 221), "tx": (171, 178, 191), "cm": (92, 99, 112)}
    for n, (kind, text) in enumerate(src):
        y = 56 + n * 28
        d.text((280, y), f"{n + 101:4d}", font=mono, fill=(75, 82, 99))
        d.text((350, y), text, font=mono, fill=colors[kind])
    img.save(path, optimize=False)


def page(path):
    img = Image.new("RGB", (1440, 900), (246, 247, 249))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1440, 64], fill=(24, 31, 52))
    big = font("DejaVuSans-Bold.ttf", 22)
    ui = font("DejaVuSans.ttf", 16)
    d.text((24, 18), "Fleet status", font=big, fill=(255, 255, 255))
    cards = [("Nodes online", "7 / 8"), ("Jobs queued", "23"), ("GPU memory", "91.4 GB"), ("Failed today", "2")]
    for i, (k, v) in enumerate(cards):
        x = 24 + i * 352
        d.rounded_rectangle([x, 88, x + 330, 188], radius=10, fill=(255, 255, 255), outline=(220, 223, 230))
        d.text((x + 18, 104), k, font=ui, fill=(110, 116, 130))
        d.text((x + 18, 134), v, font=font("DejaVuSans-Bold.ttf", 30), fill=(24, 31, 52) if i != 3 else (200, 40, 40))
    d.rounded_rectangle([24, 212, 1416, 876], radius=10, fill=(255, 255, 255), outline=(220, 223, 230))
    hdr = ["Node", "Status", "Model", "Load", "Last job"]
    xs = [48, 300, 500, 900, 1080]
    for x, h in zip(xs, hdr):
        d.text((x, 232), h, font=font("DejaVuSans-Bold.ttf", 16), fill=(60, 66, 80))
    rows = [
        ("node-a", "online", "qwen3.8-27b-nvfp4", "62%", "swe-bench #4411"),
        ("node-b", "online", "qwen3.8-27b-nvfp4", "88%", "night-17 arm B"),
        ("node-c", "offline", "-", "-", "2026-10-25 22:14"),
        ("node-d", "online", "qwen3.8-27b-minima", "12%", "refine #210"),
        ("node-e", "draining", "qwen3.8-27b-nvfp4", "40%", "index build"),
    ]
    for r, row in enumerate(rows):
        y = 272 + r * 44
        d.line([48, y - 8, 1392, y - 8], fill=(235, 237, 241))
        for x, cell in zip(xs, row):
            col = (200, 40, 40) if cell == "offline" else (210, 140, 20) if cell == "draining" else (40, 44, 52)
            d.text((x, y), cell, font=ui, fill=col)
    img.save(path, optimize=False)


def chart(path):
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ui = font("DejaVuSans.ttf", 15)
    d.text((250, 16), "Tokens per second by engine", font=font("DejaVuSans-Bold.ttf", 18), fill=(20, 20, 20))
    d.line([80, 520, 760, 520], fill=(0, 0, 0), width=2)
    d.line([80, 60, 80, 520], fill=(0, 0, 0), width=2)
    for t in range(0, 61, 10):
        y = 520 - t * 7.5
        d.line([74, y, 80, y], fill=(0, 0, 0))
        d.text((40, y - 9), str(t), font=ui, fill=(0, 0, 0))
    bars = [("A", 47.1, (66, 133, 244)), ("B", 49.3, (219, 68, 55)), ("C", 33.9, (244, 180, 0)), ("D", 52.3, (15, 157, 88))]
    for i, (name, v, c) in enumerate(bars):
        x = 140 + i * 160
        d.rectangle([x, 520 - v * 7.5, x + 90, 520], fill=c)
        d.text((x + 38, 530), name, font=ui, fill=(0, 0, 0))
        d.text((x + 25, 520 - v * 7.5 - 22), f"{v:.1f}", font=ui, fill=(0, 0, 0))
    img.save(path, optimize=False)


def icon(path):
    img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, 46, 46], fill=(250, 180, 30, 255))
    d.ellipse([14, 14, 20, 20], fill=(40, 40, 40, 255))
    d.ellipse([28, 14, 34, 20], fill=(40, 40, 40, 255))
    d.arc([12, 18, 36, 38], 20, 160, fill=(40, 40, 40, 255), width=3)
    img.save(path, optimize=False)


# The photos, from files every Ubuntu 24.04 / DGX OS install carries; checked by SHA-256 before use.
SOURCES = {
    "clouds.jpg": "Clouds_by_Tibor_Mokanszki.jpg",
    "monument.jpg": "Monument_valley_by_orbitelambda.jpg",
    "province.jpg": "Province_of_the_south_of_france_by_orbitelambda.jpg",
    "lights.webp": "Northan_lights_by_mizuno.webp",
    "station_exif.jpg": "NVIDIA_DGX_Station_Background_A.JPG",
    "grey.png": "Fuwafuwa_nanbatto_san_by_amaral-dark.png",
    "rgba.png": "desktop-wallpaper-kv-white-4k-3840-2160.png",
}


def derived(out):
    import shutil

    for name, src in SOURCES.items():
        path = "/usr/share/backgrounds/" + src
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        shutil.copyfile(path, os.path.join(out, name))
        print(f"{name} {digest} {src}")
    # As Codex sends a photo: at most 2048 px a side (here 1600 wide), JPEG at quality 85.
    for name in ("monument.jpg", "province.jpg"):
        img = Image.open(os.path.join(out, name)).convert("RGB")
        w, h = img.size
        img = img.resize((1600, round(h * 1600 / w)), Image.LANCZOS)
        small = name.replace(".jpg", "_1600.jpg")
        img.save(os.path.join(out, small), quality=85)
        print(f"{small} {hashlib.sha256(open(os.path.join(out, small), 'rb').read()).hexdigest()} (from {name})")


if __name__ == "__main__":
    what, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if what == "rendered":
        for name, fn in [("terminal.png", terminal), ("code.png", code), ("dashboard.png", page), ("chart.png", chart),
                         ("icon.png", icon)]:
            fn(os.path.join(out, name))
            print(name, hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest())
    else:
        derived(out)
