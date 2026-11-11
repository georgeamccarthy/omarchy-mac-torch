"""Render docs/<model>.mp4 (1280x720, 30 fps, H.264) and a poster frame docs/<model>.png for AlexNet, ResNet-18 and
MobileNetV3: the training animation drawn by tools/make_animation.py (an SVG, kept only as an intermediate), stepped
frame by frame in headless Chrome and encoded with ffmpeg.
Needs ffmpeg, a Chrome/Chromium binary and playwright: uv run --with playwright tools/make_video.py [models...]"""
import pathlib, re, shutil, subprocess, sys, tempfile
from playwright.sync_api import sync_playwright

root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "tools"))
import make_animation  # noqa: E402

FPS, W, H = 30, 1280, 720
chrome = shutil.which("google-chrome-stable") or shutil.which("chromium") or shutil.which("google-chrome")
models = sys.argv[1:] or list(make_animation.MODEL_LAYERS)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
    page = b.new_page(viewport={"width": W, "height": H})
    for model in models:
        with tempfile.TemporaryDirectory() as tmp:
            svg_path = pathlib.Path(tmp) / f"{model}.svg"
            make_animation.render(model, svg_path)
            svg = svg_path.read_text()
            T = float(re.search(r'dur="([\d.]+)s"', svg).group(1))
            html = (f'<body style="margin:0;background:#0d1117;display:flex;align-items:center;justify-content:center;height:{H}px">'
                    + svg.replace('width="940" height="460"', f'width="{W}" height="{int(W * 460 / 940)}"', 1) + "</body>")
            page.set_content(html)
            page.evaluate("document.querySelector('svg').pauseAnimations()")
            n = int(T * FPS)
            for i in range(n):
                page.evaluate(f"document.querySelector('svg').setCurrentTime({i / FPS})")
                page.screenshot(path=f"{tmp}/{i:04d}.png")
            shutil.copy(f"{tmp}/{n - 1:04d}.png", root / "docs" / f"{model}.png")  # poster: the finished race
            out = root / "docs" / f"{model}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)], check=True)
            print(f"wrote {out}: {n} frames, {T:.1f}s")
    b.close()
