# Video → Illustrated Study Guide

Generate WCAG 2.2 AA–compliant HTML study guides from instructional videos.

## What It Does

The pipeline takes an instructional video and produces a flowing HTML document that captures the visual and textual content a student needs to study — without the instructor's talking-head footage. Specifically, it:

1. **Detects scene changes** to find moments where the visual content shifts (e.g., new slide, diagram, animation step)
2. **Extracts key frames** at those moments and saves them as images
3. **Filters out instructor-only frames** using visual similarity analysis
4. **Runs OCR** to capture on-screen text from graphics and slides
5. **Parses the transcript** (VTT/SRT) and maps narration to each visual segment
6. **Generates an accessible HTML document** with proper structure, landmarks, and alt-text placeholders

The output is a single self-contained HTML file + a folder of frame images.

---

## Two-Pass Workflow

The script is designed for a **two-pass workflow** because good alt text requires human judgment:

### Pass 1: Extract and Export Metadata

```bash
python video_to_studyguide.py lecture.mp4 \
    --title "Introduction to Photosynthesis" \
    --module "Module 3 — Cell Energy" \
    --subtitle lecture.vtt \
    --export-metadata
```

This produces:
- `frames/` — extracted JPEG frames
- `study_guide_metadata.json` — one entry per frame with fields to fill in

### Human Review Step

Open `study_guide_metadata.json` and for each frame:

| Field            | What to Write                                                                 |
|------------------|-------------------------------------------------------------------------------|
| `alt_text`       | Describe what the graphic **teaches**, not just what it looks like (see below) |
| `section_title`  | A meaningful heading, e.g., "The Calvin Cycle" instead of "Segment at 5:03"   |
| `key_terms`      | Important vocabulary introduced in this segment                               |
| `math_equations` | LaTeX equations displayed in this segment (see format below)                  |
| `include`        | Set to `false` to drop frames that aren't useful (e.g., transitions)          |

### Pass 2: Generate Final HTML

```bash
python video_to_studyguide.py lecture.mp4 \
    --title "Introduction to Photosynthesis" \
    --subtitle lecture.vtt \
    --metadata study_guide_metadata.json \
    --output study_guide.html
```

---

## Command-Line Options

| Flag                 | Default               | Description                                      |
|----------------------|-----------------------|--------------------------------------------------|
| `--title`            | "Untitled Study Guide"| Document title                                   |
| `--module`           | (empty)               | Course/module name shown in header               |
| `--subtitle`         | (none)                | Path to .vtt or .srt file                        |
| `--auto-transcribe`  | off                   | Use OpenAI Whisper to generate transcript        |
| `--output`           | study_guide.html      | Output HTML path                                 |
| `--frames-dir`       | frames                | Where to save extracted images                   |
| `--threshold`        | 25.0                  | Scene-change sensitivity (lower = more frames)   |
| `--min-gap`          | 4.0                   | Minimum seconds between frames                   |
| `--max-frames`       | 30                    | Cap on total frames extracted                    |
| `--skip-ocr`         | off                   | Skip Tesseract OCR                               |
| `--export-metadata`  | off                   | Export JSON for human review                     |
| `--metadata`         | (none)                | Load reviewed JSON metadata                      |

---

## Dependencies

### Python Packages

```bash
pip install opencv-python-headless pillow pytesseract webvtt-py
```

### System Tools

- **ffmpeg** / **ffprobe** — frame extraction and video duration detection
- **tesseract** — OCR for on-screen text (optional, skip with `--skip-ocr`)
- **whisper** — auto-transcription (optional, only if `--auto-transcribe`)

---

## Writing Good Alt Text

The most important accessibility step is writing alt text that conveys the **instructional purpose** of each graphic. Screen reader users should get the same educational content as sighted students.

### Do This

> "Bar chart comparing photosynthesis rates under four light colors. Red light produces the highest rate at 12 μmol/m²/s, followed by blue at 9, green at 3, and far-red at 1. The chart demonstrates that chlorophyll absorbs red and blue wavelengths most efficiently."

### Not This

> "Bar chart with four colored bars."

### Guidelines

- **Lead with what it teaches**, then describe the visual evidence.
- **Include data values** from charts and graphs — a screen reader user can't estimate bar heights.
- **Describe spatial relationships** in diagrams: "Arrow points from A to B, indicating that A causes B."
- **For animation sequences**, each step's alt text should describe **what changed** from the previous step.
- **On-screen text** captured via OCR is already duplicated in the HTML body, so the alt text doesn't need to repeat it — focus on the visual elements instead.

---

## Adding Math Equations

The study guide uses **MathJax 3** to render LaTeX equations with built-in accessibility. MathJax automatically generates hidden MathML for screen readers and supports interactive equation exploration.

### Metadata Format

In the metadata JSON, add equations to any frame's `math_equations` array:

```json
{
  "index": 3,
  "math_equations": [
    {
      "latex": "6\\,\\text{CO}_2 + 6\\,\\text{H}_2\\text{O} \\xrightarrow{\\text{light}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\,\\text{O}_2",
      "label": "Equation — Overall Photosynthesis Reaction",
      "aria_label": "Six CO2 plus six H2O plus light energy yields one glucose plus six O2.",
      "caption": "Six molecules of carbon dioxide combine with six molecules of water to produce glucose and oxygen.",
      "inline": false
    }
  ]
}
```

| Field        | Required | Description                                                        |
|--------------|----------|--------------------------------------------------------------------|
| `latex`      | Yes      | LaTeX math string (use `\\text{}` for chemical names)              |
| `label`      | Yes      | Short heading shown above the equation                             |
| `aria_label` | Yes      | Plain-English description for screen readers — this is critical    |
| `caption`    | No       | Explanatory text shown below the equation                          |
| `inline`     | No       | `false` (default) for display mode, `true` for inline              |

### Accessibility Notes for Math

- The `aria_label` field is essential. Screen readers will read this instead of trying to parse the rendered equation. Write it the way you'd say the equation out loud: "E equals m c squared" not "E = mc^2".
- MathJax's `assistiveMml` option is enabled by default, which generates hidden MathML alongside the visual rendering. Screen readers that support MathML (like VoiceOver with Safari or NVDA with MathPlayer) can use this for more detailed exploration.
- The `tex-mml-chtml` MathJax output produces both HTML/CSS rendering (for visual display) and MathML (for assistive technology), satisfying WCAG 1.1.1 (Non-text Content) and 4.1.2 (Name, Role, Value).
- Display-mode equations (`inline: false`) are wrapped in a styled container with a label and optional caption, providing context that helps all users.

---

## WCAG 2.2 AA Compliance Checklist

The generated HTML addresses these success criteria. Items marked with ✋ require human action.

### Perceivable

| Criterion | Status | Notes |
|-----------|--------|-------|
| 1.1.1 Non-text Content | ✋ | Alt text placeholders generated; humans must fill them in |
| 1.3.1 Info and Relationships | ✅ | Proper heading hierarchy, landmarks, semantic HTML |
| 1.3.2 Meaningful Sequence | ✅ | DOM order matches visual reading order |
| 1.3.3 Sensory Characteristics | ✅ | No instructions rely solely on shape/color/position |
| 1.4.1 Use of Color | ✅ | No information conveyed by color alone |
| 1.4.3 Contrast (Minimum) | ✅ | All text meets 4.5:1 ratio (verified with design tokens) |
| 1.4.4 Resize Text | ✅ | Uses rem/em units; works up to 200% zoom |
| 1.4.5 Images of Text | ✋ | OCR text is duplicated in HTML body; math rendered via MathJax (not images); review for completeness |
| 1.4.10 Reflow | ✅ | Single-column layout reflows at 320px |
| 1.4.11 Non-text Contrast | ✅ | Borders and UI elements meet 3:1 ratio |

### Operable

| Criterion | Status | Notes |
|-----------|--------|-------|
| 2.1.1 Keyboard | ✅ | All interactive elements (links, skip link) keyboard-accessible |
| 2.4.1 Bypass Blocks | ✅ | Skip-to-content link present |
| 2.4.2 Page Titled | ✅ | Descriptive `<title>` element |
| 2.4.3 Focus Order | ✅ | Logical tab order follows DOM |
| 2.4.6 Headings and Labels | ✋ | Default headings are timestamps; reviewers should add descriptive ones |
| 2.4.7 Focus Visible | ✅ | 3px blue outline on all focusable elements |
| 2.4.11 Focus Appearance | ✅ | Focus indicator meets minimum area and contrast |

### Understandable

| Criterion | Status | Notes |
|-----------|--------|-------|
| 3.1.1 Language of Page | ✅ | `lang="en"` on `<html>` element |
| 3.1.2 Language of Parts | ✋ | If content includes non-English terms, add `lang` attributes |
| 3.2.3 Consistent Navigation | ✅ | Table of contents always in same position |

### Robust

| Criterion | Status | Notes |
|-----------|--------|-------|
| 4.1.2 Name, Role, Value | ✅ | ARIA labels on navigation and notes; MathJax generates assistive MathML with aria-label fallback |

---

## Customization

### Changing the Color Scheme

All colors are defined as CSS custom properties in `:root`. To change the accent color from green to blue, for example, modify these three variables in the generated HTML:

```css
--color-accent: #2563EB;
--color-accent-light: #DBEAFE;
--color-accent-dark: #1E3A8A;
```

### Adapting for Different Subjects

The template works well for conceptual videos. For other video types, consider:

- **Code walkthroughs**: Add a `.code-block` CSS class with monospace font and syntax highlighting
- **Math-heavy content**: Include MathJax or KaTeX for equation rendering
- **Lab demonstrations**: Use the step-sequence grid for procedural steps

---

## File Structure

```
project/
├── video_to_studyguide.py     # Main extraction + generation script
├── prototype_study_guide.html  # Sample output showing all component types
├── README.md                   # This file
├── frames/                     # (generated) Extracted JPEG frames
│   ├── frame_0001_0s.jpg
│   ├── frame_0002_32s.jpg
│   └── ...
├── study_guide.html            # (generated) Final HTML study guide
└── study_guide_metadata.json   # (generated) Metadata for human review
```
