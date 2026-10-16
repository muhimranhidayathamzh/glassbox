"""Assemble docs/demo.gif from the frames recorded by docs/demo/frames.mjs (needs Pillow).

    python docs/demo/make_gif.py docs/demo/frames docs/demo.gif

Frame files are named NNN_<scene>_<duration ms>.png. Each frame is scaled to a fixed width, placed on
a fixed-size canvas, and given the scene's caption.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, CAPTION_H = 720, 560, 34
BACKGROUND, INK = (247, 245, 240), (31, 29, 25)
CAPTIONS = {
    "generate": "1  Generate: the name guard hides names, before they are written",
    "microscope": "2  Microscope: which features made the model write “Lily”?",
    "replay": "3  Replay: loss, attention heads and samples across training checkpoints",
}


def font(size: int):
    for name in ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    caption_font = font(16)
    frames, durations = [], []
    for path in sorted(src.glob("*.png")):
        _, scene, ms = path.stem.split("_")
        img = Image.open(path).convert("RGB")
        scale = min(WIDTH / img.width, (HEIGHT - CAPTION_H) / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        canvas.paste(img, ((WIDTH - img.width) // 2, CAPTION_H))
        ImageDraw.Draw(canvas).text((16, 8), CAPTIONS[scene], fill=INK, font=caption_font)
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        durations.append(int(ms))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
