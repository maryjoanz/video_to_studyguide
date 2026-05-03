#!/usr/bin/env python3
"""
video_to_studyguide.py
======================
Extract key frames, transcript segments, and on-screen text from an
instructional video and generate a WCAG 2.2 AA–compliant HTML study guide.

Dependencies (install via pip):
    pip install opencv-python-headless pillow pytesseract scenedetect webvtt-py

System dependencies:
    - ffmpeg       (frame extraction, audio extraction)
    - tesseract    (OCR for on-screen text)
    - whisper CLI  (optional: auto-generate transcript if no subtitle file)

Usage:
    python video_to_studyguide.py input_video.mp4 \\
        --title "Introduction to Photosynthesis" \\
        --module "Module 3 — Cell Energy" \\
        --subtitle input_video.vtt \\
        --output study_guide.html

    # Without a subtitle file (uses OpenAI Whisper locally):
    python video_to_studyguide.py input_video.mp4 \\
        --title "Introduction to Photosynthesis" \\
        --auto-transcribe
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully with guidance
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    sys.exit("Missing opencv: pip install opencv-python-headless")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing Pillow: pip install Pillow")

try:
    import pytesseract
except ImportError:
    pytesseract = None  # OCR will be skipped

try:
    import webvtt
except ImportError:
    webvtt = None  # Subtitle parsing will be limited


# ===================================================================
# 1. SCENE DETECTION — find visually distinct moments
# ===================================================================

def detect_scene_changes(video_path: str, threshold: float = 30.0,
                         min_gap_sec: float = 3.0) -> list[float]:
    """
    Detect scene changes using frame-to-frame histogram difference.
    Returns a list of timestamps (seconds) where significant visual
    changes occur — these are candidate key-frame moments.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_gap_frames = int(min_gap_sec * fps)

    prev_hist = None
    scene_times = [0.0]  # always include the first frame
    frame_idx = 0
    last_scene_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and compute histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            # Bhattacharyya distance: 0 = identical, 1 = completely different
            if diff * 100 > threshold and (frame_idx - last_scene_frame) > min_gap_frames:
                timestamp = frame_idx / fps
                scene_times.append(timestamp)
                last_scene_frame = frame_idx

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return scene_times


def filter_instructor_frames(frames_dir: str, similarity_threshold: float = 0.92) -> list[str]:
    """
    Heuristic filter: if many consecutive frames look very similar
    (e.g., a talking head with little visual change), skip them.
    Keep frames that are visually distinct — likely slides, diagrams,
    or graphic overlays rather than the instructor on camera.

    Returns a list of file paths to KEEP.
    """
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if len(frame_files) < 2:
        return [str(f) for f in frame_files]

    kept = [str(frame_files[0])]
    prev_hist = None

    for fpath in frame_files:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist,
                                         cv2.HISTCMP_CORREL)
            if similarity < similarity_threshold:
                kept.append(str(fpath))

        prev_hist = hist

    return kept


# ===================================================================
# 2. FRAME EXTRACTION
# ===================================================================

def extract_frames_at_times(video_path: str, timestamps: list[float],
                            output_dir: str) -> list[dict]:
    """
    Extract a JPEG frame at each timestamp. Returns metadata dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for i, t in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"frame_{i:04d}_{int(t)}s.jpg"
        fpath = os.path.join(output_dir, fname)

        # Save at decent quality
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        h, w = frame.shape[:2]
        frames.append({
            "index": i,
            "timestamp": t,
            "timestamp_fmt": format_timestamp(t),
            "filename": fname,
            "path": fpath,
            "width": w,
            "height": h,
            "ocr_text": "",
            "alt_text": "",
        })

    cap.release()
    return frames


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ===================================================================
# 3. OCR — extract on-screen text from frames
# ===================================================================

def run_ocr_on_frames(frames: list[dict]) -> list[dict]:
    """
    Run Tesseract OCR on each extracted frame to capture
    on-screen text overlays, labels, and captions.
    """
    if pytesseract is None:
        print("⚠ pytesseract not installed — skipping OCR. "
              "Install with: pip install pytesseract")
        return frames

    for frame in frames:
        try:
            img = Image.open(frame["path"])
            text = pytesseract.image_to_string(img).strip()
            # Clean up OCR noise
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[^\S\n]+', ' ', text)
            frame["ocr_text"] = text if len(text) > 10 else ""
        except Exception as e:
            print(f"  OCR failed for {frame['filename']}: {e}")
            frame["ocr_text"] = ""

    return frames


# ===================================================================
# 4. TRANSCRIPT PARSING
# ===================================================================

def parse_vtt(vtt_path: str) -> list[dict]:
    """Parse WebVTT subtitle file into timed segments."""
    if webvtt is None:
        # Fallback: simple regex parser
        return _parse_vtt_simple(vtt_path)

    segments = []
    for caption in webvtt.read(vtt_path):
        segments.append({
            "start": _vtt_time_to_sec(caption.start),
            "end": _vtt_time_to_sec(caption.end),
            "text": caption.text.strip(),
        })
    return segments


def parse_srt(srt_path: str) -> list[dict]:
    """Parse SRT subtitle file into timed segments."""
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1]
        )
        if time_match:
            segments.append({
                "start": _srt_time_to_sec(time_match.group(1)),
                "end": _srt_time_to_sec(time_match.group(2)),
                "text": ' '.join(lines[2:]).strip(),
            })
    return segments


def _vtt_time_to_sec(t: str) -> float:
    parts = t.replace(',', '.').split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return int(parts[0]) * 60 + float(parts[1])


def _srt_time_to_sec(t: str) -> float:
    return _vtt_time_to_sec(t.replace(',', '.'))


def _parse_vtt_simple(vtt_path: str) -> list[dict]:
    """Regex-based VTT parser when webvtt lib isn't available."""
    segments = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n\n|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(content):
        segments.append({
            "start": _vtt_time_to_sec(m.group(1)),
            "end": _vtt_time_to_sec(m.group(2)),
            "text": m.group(3).strip(),
        })
    return segments


def auto_transcribe(video_path: str, output_dir: str) -> str:
    """
    Use OpenAI Whisper CLI to auto-generate a VTT transcript.
    Requires: pip install openai-whisper
    """
    vtt_path = os.path.join(output_dir, "transcript.vtt")
    print("🎤 Auto-transcribing with Whisper (this may take a few minutes)...")
    try:
        subprocess.run([
            "whisper", video_path,
            "--model", "base",
            "--output_format", "vtt",
            "--output_dir", output_dir,
        ], check=True, capture_output=True)
        # Whisper names files based on the input filename
        base = Path(video_path).stem
        generated = os.path.join(output_dir, f"{base}.vtt")
        if os.path.exists(generated):
            shutil.move(generated, vtt_path)
        return vtt_path
    except FileNotFoundError:
        print("⚠ Whisper CLI not found. Install: pip install openai-whisper")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"⚠ Whisper failed: {e}")
        return ""


def get_transcript_for_range(segments: list[dict],
                             start: float, end: float) -> str:
    """Get concatenated transcript text for a time range."""
    texts = []
    for seg in segments:
        if seg["end"] >= start and seg["start"] <= end:
            texts.append(seg["text"])
    return ' '.join(texts)


def segment_transcript_by_frames(frames: list[dict],
                                 transcript: list[dict]) -> list[dict]:
    """
    Assign transcript text to each frame based on the time range
    between the current frame and the next frame.
    """
    for i, frame in enumerate(frames):
        start = frame["timestamp"]
        if i + 1 < len(frames):
            end = frames[i + 1]["timestamp"]
        else:
            # Last frame: grab remaining transcript
            end = transcript[-1]["end"] if transcript else start + 30
        frame["transcript"] = get_transcript_for_range(
            transcript, start, end
        )
    return frames


# ===================================================================
# 5. HTML GENERATION
# ===================================================================

HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Study Guide: {title}</title>
  <meta name="description" content="Illustrated study guide for: {title}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --color-bg: #FAFAF7;
      --color-surface: #FFFFFF;
      --color-text: #1B1B18;
      --color-text-secondary: #4A4A45;
      --color-accent: #2D6A4F;
      --color-accent-light: #D8F3DC;
      --color-accent-dark: #1B4332;
      --color-border: #D4D4CF;
      --color-border-light: #E8E8E3;
      --color-focus: #2563EB;
      --font-heading: 'Merriweather', 'Georgia', serif;
      --font-body: 'Source Sans 3', 'Segoe UI', sans-serif;
      --font-mono: 'JetBrains Mono', 'Consolas', monospace;
      --measure: 72ch;
      --space-sm: 0.5rem;
      --space-md: 1rem;
      --space-lg: 1.5rem;
      --space-xl: 2.5rem;
      --space-2xl: 4rem;
      --radius: 6px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ font-size: 100%; scroll-behavior: smooth; scroll-padding-top: 5rem; }}
    body {{
      font-family: var(--font-body); font-size: 1.125rem; line-height: 1.7;
      color: var(--color-text); background: var(--color-bg);
      -webkit-font-smoothing: antialiased;
    }}
    .skip-link {{
      position: absolute; top: -100%; left: var(--space-md);
      background: var(--color-focus); color: #fff;
      padding: var(--space-sm) var(--space-md); border-radius: var(--radius);
      z-index: 1000; font-weight: 600; text-decoration: none;
    }}
    .skip-link:focus {{ top: var(--space-md); }}
    :focus-visible {{ outline: 3px solid var(--color-focus); outline-offset: 3px; border-radius: 2px; }}
    .doc-header {{
      background: var(--color-accent-dark); color: #fff;
      padding: var(--space-2xl) var(--space-lg); text-align: center;
    }}
    .doc-header__eyebrow {{
      font-size: 0.85rem; font-weight: 600; letter-spacing: 0.12em;
      text-transform: uppercase; color: #95D5B2; margin-bottom: var(--space-sm);
    }}
    .doc-header h1 {{
      font-family: var(--font-heading); font-size: clamp(1.75rem, 4vw, 2.75rem);
      font-weight: 700; line-height: 1.25; max-width: 20em; margin: 0 auto var(--space-md);
    }}
    .doc-header__meta {{ font-size: 0.95rem; color: #B7E4C7; line-height: 1.5; }}
    .doc-header__meta span + span::before {{ content: "·"; margin: 0 0.5em; }}
    .toc {{
      background: var(--color-surface); border: 1px solid var(--color-border);
      border-radius: var(--radius); max-width: 40rem;
      margin: var(--space-xl) auto; padding: var(--space-lg) var(--space-xl);
    }}
    .toc h2 {{ font-family: var(--font-heading); font-size: 1.15rem; margin-bottom: var(--space-md); color: var(--color-accent-dark); }}
    .toc ol {{ list-style: decimal; padding-left: 1.5em; }}
    .toc li {{ margin-bottom: var(--space-sm); }}
    .toc a {{ color: var(--color-text); text-decoration: underline; text-underline-offset: 3px; }}
    .toc a:hover {{ color: var(--color-accent-dark); }}
    .content {{ max-width: var(--measure); margin: 0 auto; padding: var(--space-lg); }}
    .section {{ margin-bottom: var(--space-2xl); padding-bottom: var(--space-xl); border-bottom: 1px solid var(--color-border-light); }}
    .section:last-child {{ border-bottom: none; }}
    .section h2 {{ font-family: var(--font-heading); font-size: 1.65rem; color: var(--color-accent-dark); margin-bottom: var(--space-md); line-height: 1.3; }}
    .section p {{ margin-bottom: var(--space-md); }}
    .figure {{
      margin: var(--space-xl) 0; background: var(--color-surface);
      border: 1px solid var(--color-border); border-radius: var(--radius); overflow: hidden;
    }}
    .figure img {{ display: block; width: 100%; height: auto; }}
    .figure figcaption {{
      padding: var(--space-sm) var(--space-md); font-size: 0.9rem;
      color: var(--color-text-secondary); border-top: 1px solid var(--color-border-light); line-height: 1.5;
    }}
    .figure__timestamp {{
      display: inline-block; font-family: var(--font-mono); font-size: 0.78rem;
      background: var(--color-accent-light); color: var(--color-accent-dark);
      padding: 0.1em 0.5em; border-radius: 3px; margin-right: 0.4em;
    }}
    .onscreen-text {{
      margin: var(--space-lg) 0; padding: var(--space-md) var(--space-lg);
      border-left: 4px solid var(--color-accent); background: var(--color-accent-light);
      border-radius: 0 var(--radius) var(--radius) 0;
    }}
    .onscreen-text__label {{
      display: block; font-size: 0.78rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.08em;
      color: var(--color-accent-dark); margin-bottom: 0.25rem;
    }}
    .math-block {{
      margin: var(--space-lg) 0; padding: var(--space-lg);
      background: var(--color-surface); border: 1px solid var(--color-border);
      border-radius: var(--radius); text-align: center; overflow-x: auto;
    }}
    .math-block__label {{
      display: block; font-size: 0.78rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.08em;
      color: var(--color-accent-dark); margin-bottom: var(--space-sm); text-align: left;
    }}
    .math-block__caption {{
      display: block; font-size: 0.88rem; color: var(--color-text-secondary);
      margin-top: var(--space-sm); text-align: left; line-height: 1.5;
    }}
    .math-block mjx-container {{ color: var(--color-text) !important; font-size: 1.2em !important; }}
    .doc-footer {{
      text-align: center; padding: var(--space-xl) var(--space-lg);
      font-size: 0.85rem; color: var(--color-text-secondary);
      border-top: 1px solid var(--color-border-light);
      max-width: var(--measure); margin: 0 auto;
    }}
    @media print {{
      body {{ font-size: 11pt; background: #fff; }}
      .doc-header {{ background: none; color: var(--color-text); padding: 1em 0; border-bottom: 2px solid #000; }}
      .skip-link {{ display: none; }}
      .section, .figure {{ page-break-inside: avoid; }}
    }}
    @media (max-width: 600px) {{
      .content {{ padding: var(--space-md); }}
      .toc {{ margin: var(--space-md); padding: var(--space-md); }}
    }}
  </style>

  <!-- MathJax 3 with accessibility (assistive MathML for screen readers) -->
  <script>
    MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
      }},
      options: {{
        enableMenu: true,
        menuOptions: {{
          settings: {{
            assistiveMml: true,
            explorer: true,
          }}
        }}
      }}
    }};
  </script>
  <script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>

  <header class="doc-header" role="banner">
    <p class="doc-header__eyebrow">Illustrated Study Guide</p>
    <h1>{title}</h1>
    <p class="doc-header__meta">
      <span>{module}</span>
      <span>Source video: {duration}</span>
      <span>Generated: {date}</span>
    </p>
  </header>

  <nav class="toc" aria-label="Table of contents">
    <h2>Contents</h2>
    <ol>
{toc_entries}
    </ol>
  </nav>

  <main id="main-content" class="content">
{sections}
  </main>

  <footer class="doc-footer" role="contentinfo">
    <p>
      This study guide was auto-generated from video content.
      Frames were extracted at key visual transitions. On-screen text was captured via OCR.
      Transcript content may require review for accuracy.
    </p>
    <p>
      <strong>Accessibility:</strong> This document targets WCAG 2.2 Level AA.
      All images require descriptive alt text — placeholders marked with
      <code>[ALT TEXT NEEDED]</code> should be replaced before publishing.
    </p>
  </footer>
</body>
</html>
""")


def build_math_block(equation: dict) -> str:
    """
    Build HTML for a MathJax-rendered equation block.

    equation dict should contain:
        latex:       LaTeX string (e.g., "E = mc^2")
        label:       Short label (e.g., "Equation — Energy-Mass Equivalence")
        aria_label:  Plain-English description for screen readers
        caption:     Optional explanatory caption below the equation
        inline:      If True, render inline \\(...\\); if False, display \\[...\\]
    """
    latex = equation.get("latex", "")
    label = _escape(equation.get("label", "Equation"))
    aria = _escape(equation.get("aria_label", f"Mathematical equation: {label}"))
    caption = equation.get("caption", "")
    inline = equation.get("inline", False)

    if not latex:
        return ""

    if inline:
        rendered = f"\\({latex}\\)"
    else:
        rendered = f"\\[\n          {latex}\n        \\]"

    caption_html = ""
    if caption:
        caption_html = (
            f'\n        <span class="math-block__caption">'
            f'{_escape(caption)}</span>'
        )

    return textwrap.dedent(f"""\

      <div class="math-block" role="math" aria-label="{aria}">
        <span class="math-block__label">{label}</span>
        {rendered}{caption_html}
      </div>""")


def build_section_html(idx: int, frame: dict, frames_rel_dir: str) -> str:
    """Build HTML for a single study-guide section from a frame."""
    section_id = f"section-{idx}"
    heading_id = f"heading-{idx}"
    ts = frame["timestamp_fmt"]

    # Alt text placeholder — must be filled by a human reviewer
    alt = frame.get("alt_text") or "[ALT TEXT NEEDED: Describe what this graphic teaches, not just what it shows.]"

    # Build the image figure
    img_path = f"{frames_rel_dir}/{frame['filename']}"
    figure_html = textwrap.dedent(f"""\
      <figure class="figure">
        <img
          src="{img_path}"
          alt="{_escape(alt)}"
          width="{frame['width']}" height="{frame['height']}"
          loading="lazy"
        >
        <figcaption>
          <span class="figure__timestamp" aria-label="Video timestamp">{ts}</span>
          Frame extracted at {ts} in the source video.
        </figcaption>
      </figure>""")

    # On-screen text (from OCR)
    ocr_block = ""
    if frame.get("ocr_text"):
        ocr_block = textwrap.dedent(f"""\

      <div class="onscreen-text" role="note">
        <span class="onscreen-text__label">On-screen text from video</span>
        {_escape(frame["ocr_text"])}
      </div>""")

    # Transcript narration
    narration = ""
    if frame.get("transcript"):
        narration = f"\n\n      <p>{_escape(frame['transcript'])}</p>"

    # Math equations (from metadata)
    math_html = ""
    for eq in frame.get("math_equations", []):
        math_html += build_math_block(eq)

    return textwrap.dedent(f"""\
    <section class="section" id="{section_id}" aria-labelledby="{heading_id}">
      <h2 id="{heading_id}">{idx + 1}. Segment at {ts}</h2>
{figure_html}{ocr_block}{math_html}{narration}
    </section>
""")


def build_toc_entry(idx: int, frame: dict) -> str:
    ts = frame["timestamp_fmt"]
    return f'      <li><a href="#section-{idx}">Segment at {ts}</a></li>'


def get_video_duration(video_path: str) -> str:
    """Get formatted video duration via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        secs = float(result.stdout.strip())
        m, s = divmod(int(secs), 60)
        return f"{m} min {s:02d} sec"
    except Exception:
        return "unknown duration"


def generate_html(frames: list[dict], title: str, module: str,
                  video_path: str, frames_rel_dir: str) -> str:
    """Assemble the final HTML document."""
    toc = "\n".join(build_toc_entry(i, f) for i, f in enumerate(frames))
    sections = "\n".join(
        build_section_html(i, f, frames_rel_dir) for i, f in enumerate(frames)
    )
    duration = get_video_duration(video_path) if video_path else "—"

    return HTML_TEMPLATE.format(
        title=_escape(title),
        module=_escape(module),
        duration=duration,
        date=datetime.now().strftime("%B %Y"),
        toc_entries=toc,
        sections=sections,
    )


def _escape(text: str) -> str:
    """Basic HTML escaping."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ===================================================================
# 6. METADATA EXPORT (for alt-text authoring workflow)
# ===================================================================

def export_metadata(frames: list[dict], output_path: str):
    """
    Export a JSON file that a human reviewer can use to add alt text,
    section titles, and key-term definitions. This can then be re-fed
    into the script to produce a polished final guide.
    """
    meta = []
    for f in frames:
        meta.append({
            "index": f["index"],
            "timestamp": f["timestamp_fmt"],
            "filename": f["filename"],
            "ocr_text": f["ocr_text"],
            "transcript_snippet": (f.get("transcript", "")[:200] + "..."
                                   if len(f.get("transcript", "")) > 200
                                   else f.get("transcript", "")),
            "alt_text": "",           # ← FILL THIS IN
            "section_title": "",      # ← FILL THIS IN (optional)
            "key_terms": [],          # ← ADD TERMS (optional)
            "math_equations": [],     # ← ADD EQUATIONS (optional, see format below)
            # math_equations format: [{"latex": "E = mc^2",
            #   "label": "Equation — Energy-Mass Equivalence",
            #   "aria_label": "E equals m c squared",
            #   "caption": "Explains the relationship between energy and mass.",
            #   "inline": false}]
            "is_animation_step": False,  # ← SET TRUE to group as sequence
            "include": True,          # ← SET FALSE to exclude this frame
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"📝 Metadata exported to: {output_path}")
    print("   → Fill in alt_text, section_title, and key_terms")
    print("   → Add math_equations with LaTeX + aria_label for accessibility")
    print("   → Set include=false to exclude frames")
    print("   → Re-run with --metadata to produce final guide")


def load_metadata(meta_path: str, frames: list[dict]) -> list[dict]:
    """Merge human-authored metadata back into frame data."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_by_idx = {m["index"]: m for m in meta}

    result = []
    for frame in frames:
        m = meta_by_idx.get(frame["index"], {})
        if m.get("include", True) is False:
            continue  # Skip excluded frames
        if m.get("alt_text"):
            frame["alt_text"] = m["alt_text"]
        if m.get("section_title"):
            frame["section_title"] = m["section_title"]
        if m.get("math_equations"):
            frame["math_equations"] = m["math_equations"]
        result.append(frame)

    return result


# ===================================================================
# 7. MAIN PIPELINE
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a WCAG 2.2 AA HTML study guide from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Workflow:
          1) Run with --export-metadata to get a JSON file for review
          2) Fill in alt_text and section_title in the JSON
          3) Re-run with --metadata to produce the final HTML

        Example:
          python video_to_studyguide.py lecture.mp4 \\
              --title "Intro to Photosynthesis" \\
              --subtitle lecture.vtt \\
              --export-metadata

          # ... edit metadata.json ...

          python video_to_studyguide.py lecture.mp4 \\
              --title "Intro to Photosynthesis" \\
              --subtitle lecture.vtt \\
              --metadata metadata.json \\
              --output study_guide.html
        """)
    )

    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--title", default="Untitled Study Guide",
                        help="Title for the study guide")
    parser.add_argument("--module", default="",
                        help="Module or course name")
    parser.add_argument("--subtitle", default=None,
                        help="Path to .vtt or .srt subtitle file")
    parser.add_argument("--auto-transcribe", action="store_true",
                        help="Auto-generate transcript with Whisper")
    parser.add_argument("--output", default="study_guide.html",
                        help="Output HTML file path")
    parser.add_argument("--frames-dir", default="frames",
                        help="Directory for extracted frames")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="Scene-change sensitivity (lower = more frames)")
    parser.add_argument("--min-gap", type=float, default=4.0,
                        help="Minimum seconds between extracted frames")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Maximum number of frames to extract")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="Skip OCR on extracted frames")
    parser.add_argument("--export-metadata", action="store_true",
                        help="Export metadata JSON for human review")
    parser.add_argument("--metadata", default=None,
                        help="Path to reviewed metadata JSON")

    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        sys.exit(f"Video file not found: {video_path}")

    print(f"🎬 Processing: {video_path}")

    # --- Step 1: Detect scenes ---
    print("🔍 Detecting scene changes...")
    scene_times = detect_scene_changes(
        video_path, threshold=args.threshold, min_gap_sec=args.min_gap
    )
    print(f"   Found {len(scene_times)} candidate scene changes")

    # Cap at max_frames
    if len(scene_times) > args.max_frames:
        # Keep evenly spaced subset
        step = len(scene_times) / args.max_frames
        scene_times = [scene_times[int(i * step)]
                       for i in range(args.max_frames)]

    # --- Step 2: Extract frames ---
    print(f"🖼  Extracting {len(scene_times)} frames...")
    frames = extract_frames_at_times(video_path, scene_times, args.frames_dir)
    print(f"   Saved to: {args.frames_dir}/")

    # --- Step 3: Filter out instructor-only frames ---
    print("🧹 Filtering similar/instructor frames...")
    kept_paths = set(filter_instructor_frames(args.frames_dir))
    before = len(frames)
    frames = [f for f in frames if f["path"] in kept_paths]
    print(f"   Kept {len(frames)} of {before} frames")

    # --- Step 4: OCR ---
    if not args.skip_ocr:
        print("🔤 Running OCR on frames...")
        frames = run_ocr_on_frames(frames)
        ocr_count = sum(1 for f in frames if f["ocr_text"])
        print(f"   Found on-screen text in {ocr_count} frames")

    # --- Step 5: Transcript ---
    transcript = []
    sub_path = args.subtitle

    if sub_path is None and args.auto_transcribe:
        sub_path = auto_transcribe(video_path, args.frames_dir)

    if sub_path and os.path.exists(sub_path):
        print(f"📜 Parsing transcript: {sub_path}")
        if sub_path.endswith(".vtt"):
            transcript = parse_vtt(sub_path)
        elif sub_path.endswith(".srt"):
            transcript = parse_srt(sub_path)
        else:
            print(f"   ⚠ Unsupported subtitle format: {sub_path}")

        if transcript:
            frames = segment_transcript_by_frames(frames, transcript)
            print(f"   Mapped {len(transcript)} subtitle segments to frames")
    else:
        print("⚠ No transcript available — narration text will be empty")

    # --- Step 6a: Export metadata for review ---
    if args.export_metadata:
        meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
        export_metadata(frames, meta_path)
        print("\n✅ Metadata exported. Edit it, then re-run with --metadata")
        return

    # --- Step 6b: Load reviewed metadata ---
    if args.metadata and os.path.exists(args.metadata):
        print(f"📋 Loading reviewed metadata: {args.metadata}")
        frames = load_metadata(args.metadata, frames)

    # --- Step 7: Generate HTML ---
    print("📄 Generating HTML study guide...")
    html = generate_html(
        frames=frames,
        title=args.title,
        module=args.module,
        video_path=video_path,
        frames_rel_dir=args.frames_dir,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Study guide saved to: {args.output}")
    print(f"   {len(frames)} sections with frames")

    alt_needed = sum(1 for f in frames if "[ALT TEXT NEEDED" in f.get("alt_text", "[ALT"))
    if alt_needed:
        print(f"\n⚠  {alt_needed} images need alt text before publishing.")
        print("   Run with --export-metadata, fill in alt_text, then re-run with --metadata.")


if __name__ == "__main__":
    main()
