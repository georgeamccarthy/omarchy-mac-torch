"""Render docs/alexnet.mp4 (1280x720, 30 fps, H.264) from docs/alexnet.svg for posting on social media.
Needs ffmpeg, a Chrome/Chromium binary and playwright: uv run --with playwright tools/make_video.py"""
import pathlib, re, shutil, subprocess, tempfile
from playwright.sync_api import sync_playwright

root = pathlib.Path(__file__).resolve().parent.parent
svg = (root / "docs" / "alexnet.svg").read_text()
T = float(re.search(r'dur="([\d.]+)s"', svg).group(1))
FPS, W, H = 30, 1280, 720
chrome = shutil.which("google-chrome-stable") or shutil.which("chromium") or shutil.which("google-chrome")
html = (f'<body style="margin:0;background:#0d1117;display:flex;align-items:center;justify-content:center;height:{H}px">'
        + svg.replace('width="940" height="460"', f'width="{W}" height="{int(W * 460 / 940)}"', 1) + "</body>")
out = root / "docs" / "alexnet.mp4"
with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
    page = b.new_page(viewport={"width": W, "height": H})
    page.set_content(html)
    page.evaluate("document.querySelector('svg').pauseAnimations()")
    n = int(T * FPS)
    for i in range(n):
        page.evaluate(f"document.querySelector('svg').setCurrentTime({i / FPS})")
        page.screenshot(path=f"{tmp}/{i:04d}.png")
    b.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)], check=True)
print(f"wrote {out}: {n} frames, {T:.1f}s")
