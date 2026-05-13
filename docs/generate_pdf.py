#!/usr/bin/env python3
"""Generate the Paryaya RunPod fine-tuning guide as a PDF."""
from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Colours ──────────────────────────────────────────────────────────────────
C_NAVY     = colors.HexColor("#0F172A")
C_BLUE     = colors.HexColor("#3B82F6")
C_TEAL     = colors.HexColor("#0EA5E9")
C_GREEN    = colors.HexColor("#10B981")
C_AMBER    = colors.HexColor("#F59E0B")
C_RED      = colors.HexColor("#EF4444")
C_CODE_BG  = colors.HexColor("#1E293B")
C_CODE_FG  = colors.HexColor("#E2E8F0")
C_RULE     = colors.HexColor("#334155")
C_MUTED    = colors.HexColor("#64748B")
C_INFO_BG  = colors.HexColor("#EFF6FF")
C_INFO_BD  = colors.HexColor("#3B82F6")
C_WARN_BG  = colors.HexColor("#FFFBEB")
C_WARN_BD  = colors.HexColor("#F59E0B")
C_TIP_BG   = colors.HexColor("#F0FDF4")
C_TIP_BD   = colors.HexColor("#10B981")
C_PAGE_BG  = colors.white

W, H = A4
MARGIN_L = 20 * mm
MARGIN_R = 20 * mm
MARGIN_T = 22 * mm
MARGIN_B = 22 * mm
TW = W - MARGIN_L - MARGIN_R          # text width


# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw) -> ParagraphStyle:
    parent = kw.pop("parent", "Normal")
    ps = ParagraphStyle(name, parent=base[parent], **kw)
    return ps

TITLE1   = S("Title1",  fontSize=28, leading=34, textColor=C_NAVY,
             fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
SUBTITLE = S("Subtitle", fontSize=12, leading=18, textColor=C_MUTED,
             fontName="Helvetica", spaceAfter=3, alignment=TA_CENTER)
BADGE    = S("Badge", fontSize=9, leading=13, textColor=colors.white,
             fontName="Helvetica-Bold", alignment=TA_CENTER)
H1       = S("H1", fontSize=16, leading=22, textColor=colors.white,
             fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=14,
             backColor=C_NAVY, leftIndent=-2, rightIndent=-2,
             borderPad=(6, 10, 6, 10))
H2       = S("H2", fontSize=12, leading=18, textColor=C_NAVY,
             fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10,
             borderPad=0)
H3       = S("H3", fontSize=10, leading=15, textColor=C_TEAL,
             fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=7)
BODY     = S("Body", fontSize=9, leading=14, textColor=C_NAVY,
             fontName="Helvetica", spaceAfter=4)
BODY_SM  = S("BodySm", fontSize=8, leading=12, textColor=C_NAVY,
             fontName="Helvetica", spaceAfter=3)
BULLET   = S("Bullet", fontSize=9, leading=14, textColor=C_NAVY,
             fontName="Helvetica", leftIndent=14, firstLineIndent=-10,
             spaceAfter=3)
BULLET2  = S("Bullet2", fontSize=9, leading=14, textColor=C_NAVY,
             fontName="Helvetica", leftIndent=28, firstLineIndent=-10,
             spaceAfter=2)
CODE     = S("Code", fontSize=8, leading=12, textColor=C_CODE_FG,
             fontName="Courier", backColor=C_CODE_BG,
             leftIndent=8, rightIndent=8, spaceAfter=6, spaceBefore=2,
             borderPad=(6, 8, 6, 8))
NOTE     = S("Note", fontSize=8.5, leading=13, textColor=C_NAVY,
             fontName="Helvetica-Oblique", leftIndent=12)
CAPTION  = S("Caption", fontSize=7.5, leading=11, textColor=C_MUTED,
             fontName="Helvetica-Oblique", alignment=TA_CENTER)
TH       = S("TH", fontSize=8, leading=12, textColor=colors.white,
             fontName="Helvetica-Bold", alignment=TA_CENTER)
TD       = S("TD", fontSize=8, leading=12, textColor=C_NAVY,
             fontName="Helvetica")
TD_CODE  = S("TDCode", fontSize=7.5, leading=11, textColor=C_NAVY,
             fontName="Courier")
STEP_NUM = S("StepNum", fontSize=22, leading=28, textColor=C_BLUE,
             fontName="Helvetica-Bold", alignment=TA_CENTER)


# ── Page template ─────────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    page = doc.page

    # Top rule
    canvas.setStrokeColor(C_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, H - MARGIN_T + 4*mm, W - MARGIN_R, H - MARGIN_T + 4*mm)

    # Header text
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN_L, H - MARGIN_T + 5.5*mm, "PARYAYA — RunPod Whisper Fine-Tuning Guide")
    canvas.drawRightString(W - MARGIN_R, H - MARGIN_T + 5.5*mm, "github.com/asticrat/paryaya")

    # Bottom rule
    canvas.line(MARGIN_L, MARGIN_B - 4*mm, W - MARGIN_R, MARGIN_B - 4*mm)

    # Page number
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W / 2, MARGIN_B - 7*mm, f"— {page} —")
    canvas.restoreState()


def _cover_page(canvas, doc):
    # Solid navy background
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Accent bar
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, H * 0.38, W, 4, fill=1, stroke=0)

    # Decorative dots (grid)
    canvas.setFillColor(colors.HexColor("#1E293B"))
    for x in range(int(MARGIN_L), int(W - MARGIN_R), 18):
        for y in range(int(MARGIN_B), int(H - MARGIN_B), 18):
            canvas.circle(x, y, 1, fill=1, stroke=0)

    # Cost / time badge strip
    canvas.setFillColor(C_BLUE)
    canvas.roundRect(MARGIN_L, H*0.33, TW, 26, 6, fill=1, stroke=0)


def build_cover() -> list:
    items = []
    items.append(Spacer(1, 52*mm))
    items.append(Paragraph("PARYAYA", S("_cn", fontSize=11, leading=14,
        textColor=C_TEAL, fontName="Helvetica-Bold", alignment=TA_CENTER,
        spaceAfter=2)))
    items.append(Paragraph("पर्याय", S("_np", fontSize=14, leading=18,
        textColor=colors.HexColor("#94A3B8"), fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=8)))
    items.append(Paragraph("RunPod Whisper<br/>Fine-Tuning Guide",
        S("_t", fontSize=34, leading=42, textColor=colors.white,
          fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)))
    items.append(Paragraph(
        "Complete step-by-step: accounts → GPU pod → train → download → deploy",
        S("_s", fontSize=11, leading=16, textColor=colors.HexColor("#94A3B8"),
          fontName="Helvetica", alignment=TA_CENTER, spaceAfter=22)))

    # Badge row  (cost | time)
    badge_data = [
        [Paragraph("~$10–20 total cost", TH),
         Paragraph("~3–6 hrs training (unattended)", TH),
         Paragraph("A100 PCIe 40 GB", TH)],
    ]
    badge_table = Table(badge_data, colWidths=[TW*0.33]*3)
    badge_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,-1), colors.white),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
        ("LINEBEFORE",   (1,0), (1,-1), 1, colors.HexColor("#1E40AF")),
        ("LINEBEFORE",   (2,0), (2,-1), 1, colors.HexColor("#1E40AF")),
        ("ROUNDEDCORNERS", [6]),
    ]))
    items.append(badge_table)
    items.append(Spacer(1, 14*mm))

    # What this guide covers box
    items.append(info_box(
        "What This Guide Covers",
        [
            "Fine-tune OpenAI Whisper medium (244M params) on Google FLEURS Nepali (ne_np)",
            "FLEURS is free and public — no account, no token, no terms acceptance needed",
            "No custom model architecture — uses HuggingFace Transformers Seq2SeqTrainer",
            "After training: point Paryaya API at checkpoint with ASR_BACKEND=whisper",
        ],
        bg=colors.HexColor("#0F2044"), border=C_BLUE, text_color=colors.white,
        title_color=C_TEAL,
    ))
    items.append(Spacer(1, 10*mm))
    items.append(Paragraph(
        "Before: ASR_BACKEND=paryaya → custom conformer (untrained)  |  "
        "After: ASR_BACKEND=whisper → Whisper medium fine-tuned on real Nepali",
        S("_flow", fontSize=8, leading=12, textColor=colors.HexColor("#64748B"),
          fontName="Courier", alignment=TA_CENTER)))
    items.append(PageBreak())
    return items


# ── Reusable block builders ───────────────────────────────────────────────────
def rule() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=C_RULE,
                      spaceAfter=6, spaceBefore=2)

def spacer(h: float = 4) -> Spacer:
    return Spacer(1, h*mm)

def h1(text: str) -> list:
    """Part header — full-width navy bar."""
    return [
        spacer(3),
        Table([[Paragraph(text, H1)]], colWidths=[TW],
              style=[
                  ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
                  ("TOPPADDING", (0,0), (-1,-1), 8),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                  ("LEFTPADDING", (0,0), (-1,-1), 10),
                  ("RIGHTPADDING", (0,0), (-1,-1), 10),
              ]),
        spacer(2),
    ]

def h2(text: str) -> list:
    return [
        spacer(1),
        Paragraph(text, H2),
        HRFlowable(width="100%", thickness=1.5, color=C_BLUE,
                   spaceAfter=4, spaceBefore=1),
    ]

def h3(text: str) -> list:
    return [Paragraph(text, H3)]

def body(text: str) -> Paragraph:
    return Paragraph(text, BODY)

def bullet(text: str, level: int = 1) -> Paragraph:
    style = BULLET if level == 1 else BULLET2
    prefix = "•" if level == 1 else "◦"
    return Paragraph(f"{prefix}  {text}", style)

def code_block(text: str) -> Preformatted:
    return Preformatted(text.rstrip(), CODE)

def note(text: str) -> Paragraph:
    return Paragraph(f"<i>Note: {text}</i>", NOTE)

def _hex(c) -> str:
    try:
        return c.hexval()[2:]
    except Exception:
        return "3B82F6"


def info_box(title: str, points: list[str], bg=C_INFO_BG, border=C_INFO_BD,
             text_color=C_NAVY, title_color=None) -> Table:
    if title_color is None:
        title_color = border
    hex_color = _hex(title_color)
    content = [Paragraph(f"<b><font color='#{hex_color}'>{title}</font></b>",
                          S("_ibh", fontSize=9, leading=13, fontName="Helvetica-Bold",
                            textColor=title_color, spaceAfter=4))]
    for p in points:
        content.append(Paragraph(f"•  {p}",
            S("_ibp", fontSize=8.5, leading=13, fontName="Helvetica",
              textColor=text_color, leftIndent=8, spaceAfter=2)))
    t = Table([[content]], colWidths=[TW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINEBEFORE",    (0,0), (0,-1), 3, border),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t

def warn_box(text: str) -> Table:
    content = [Paragraph(f"⚠️  <b>Warning:</b>  {text}",
        S("_wb", fontSize=8.5, leading=13, fontName="Helvetica", textColor=C_NAVY))]
    t = Table([[content]], colWidths=[TW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_WARN_BG),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LINEBEFORE",    (0,0), (0,-1), 3, C_AMBER),
    ]))
    return t

def tip_box(text: str) -> Table:
    content = [Paragraph(f"✅  {text}",
        S("_tb", fontSize=8.5, leading=13, fontName="Helvetica", textColor=C_NAVY))]
    t = Table([[content]], colWidths=[TW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_TIP_BG),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LINEBEFORE",    (0,0), (0,-1), 3, C_GREEN),
    ]))
    return t

def data_table(headers: list[str], rows: list[list[str]],
               col_widths: list[float] | None = None) -> Table:
    if col_widths is None:
        col_widths = [TW / len(headers)] * len(headers)

    header_row = [Paragraph(h, TH) for h in headers]
    data_rows  = []
    for i, row in enumerate(rows):
        style = TD_CODE if any(c in " ".join(row) for c in ("export ", "pip ", "docker", "python", "git", "ssh", "curl")) else TD
        data_rows.append([Paragraph(str(c), style) for c in row])

    table_data = [header_row] + data_rows
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("LINEBELOW",     (0,0), (-1,0), 1, C_BLUE),
    ]))
    return t

def step_box(number: str, title: str, time: str) -> Table:
    """Numbered step header with time badge."""
    num_cell  = Paragraph(number, STEP_NUM)
    title_cell = [
        Paragraph(title, S("_st", fontSize=13, leading=18, fontName="Helvetica-Bold",
                            textColor=C_NAVY, spaceAfter=2)),
        Paragraph(time, S("_stt", fontSize=8, leading=11, fontName="Helvetica",
                           textColor=C_MUTED)),
    ]
    badge_cell = Table([[Paragraph(time, S("_sb", fontSize=8, leading=11,
                                            fontName="Helvetica-Bold",
                                            textColor=colors.white, alignment=TA_CENTER))]],
                       colWidths=[28*mm])
    badge_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_TEAL),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))

    outer = Table([[num_cell, title_cell, badge_cell]],
                  colWidths=[18*mm, TW - 18*mm - 30*mm, 30*mm])
    outer.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (0,-1), 0),
        ("RIGHTPADDING",  (-1,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEBEFORE",    (0,0), (0,-1), 3, C_BLUE),
        ("LINEBELOW",     (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    return outer


# ── Content ───────────────────────────────────────────────────────────────────
def build_toc() -> list:
    items = []
    items += h1("TABLE OF CONTENTS")
    toc = [
        ("PART 0", "Before You Start — Accounts & Keys",      "3"),
        ("PART 1", "Create RunPod Account and Launch Pod",    "5"),
        ("PART 2", "Set Up the Pod",                          "7"),
        ("PART 3", "Smoke Test",                              "9"),
        ("PART 4", "Full Training",                           "10"),
        ("PART 5", "Download Model and Terminate Pod",        "13"),
        ("PART 6", "Deploy via Paryaya API",                  "15"),
        ("",       "Troubleshooting",                         "16"),
        ("",       "Quick Reference",                         "19"),
        ("",       "Cost Summary",                            "20"),
    ]
    for part, title, page in toc:
        row = f"<b>{part}</b>  —  {title}" if part else f"        {title}"
        items.append(
            Table([[Paragraph(row, BODY), Paragraph(page, S("_pg", fontSize=9, leading=14,
                    fontName="Helvetica", textColor=C_MUTED, alignment=TA_RIGHT))]],
                  colWidths=[TW - 12*mm, 12*mm],
                  style=[("LINEBELOW", (0,0), (-1,-1), 0.3,
                          colors.HexColor("#E2E8F0")),
                         ("TOPPADDING", (0,0), (-1,-1), 4),
                         ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    items.append(PageBreak())
    return items


def build_part0() -> list:
    items = []
    items += h1("PART 0 — BEFORE YOU START")
    items.append(body("Complete these two steps before launching a RunPod pod."))
    items.append(spacer(2))
    items.append(tip_box(
        "Good news: we use Google FLEURS (ne_np) as the training dataset. "
        "FLEURS is completely free and public — no HuggingFace account, no token, "
        "and no terms acceptance required. Just run the training script and it downloads automatically."))
    items.append(spacer(3))

    # Step 0.1
    items.append(step_box("0.1", "Create a Weights & Biases Account", "~5 minutes (optional)"))
    items.append(spacer(2))
    items.append(info_box("Why FLEURS instead of Common Voice?", [
        "Mozilla moved all Common Voice datasets off HuggingFace in October 2025",
        "Google FLEURS (ne_np) is still on HuggingFace, free, and requires no login",
        "FLEURS has ~3,000 high-quality Nepali training clips — sufficient for fine-tuning",
        "Training takes 2-3 hours on A100 vs 6-8 hours for Common Voice (smaller dataset)",
    ]))
    items.append(spacer(2))

    items.append(step_box("0.1", "Create a Weights & Biases Account", "~5 minutes (optional)"))
    items.append(spacer(2))
    items.append(tip_box(
        "W&B gives you real-time loss and WER charts you can watch from your phone "
        "without keeping SSH open. Free tier is enough. Skip only if you want no monitoring."))
    items.append(spacer(2))
    items.append(bullet("Go to <b>wandb.ai</b> and click <b>Sign up</b>"))
    items.append(bullet("Once logged in, go to <b>wandb.ai/settings</b>"))
    items.append(bullet("Scroll to <b>API keys</b> → click <b>New key</b> → copy it"))
    items.append(body("Your W&B key looks like:"))
    items.append(code_block("abcdef1234567890abcdef1234567890abcdef12"))
    items.append(spacer(3))

    items.append(step_box("0.2", "Verify Your SSH Key Exists", "~5 minutes"))
    items.append(spacer(2))
    items.append(body("You need an SSH key to connect to RunPod. Check if you have one:"))
    items.append(code_block("# Run on your Mac terminal:\nls ~/.ssh/id_ed25519.pub"))
    items.append(body("<b>If the file exists</b> — skip to \"Copy your public key\" below."))
    items.append(body("<b>If you get \"No such file or directory\"</b> — create one:"))
    items.append(code_block(
        "ssh-keygen -t ed25519 -C \"paryaya-runpod\"\n"
        "# Press Enter three times (accept defaults, no passphrase)"))
    items += h3("Copy Your Public Key")
    items.append(body("You will paste this into RunPod in Part 1:"))
    items.append(code_block("cat ~/.ssh/id_ed25519.pub\n# Select all output and copy it"))
    items.append(body("The output looks like:"))
    items.append(code_block(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... paryaya-runpod"))
    items.append(PageBreak())
    return items


def build_part1() -> list:
    items = []
    items += h1("PART 1 — CREATE RUNPOD ACCOUNT AND LAUNCH GPU POD")
    items.append(Paragraph("~20 minutes — do this once", S("_tm", fontSize=9, leading=13,
                            fontName="Helvetica-Oblique", textColor=C_MUTED, spaceAfter=8)))

    items.append(step_box("1.1", "Create RunPod Account and Add Credits", "~5 minutes"))
    items.append(spacer(2))
    items.append(bullet("Go to <b>runpod.io</b> and click <b>Sign Up</b>"))
    items.append(bullet("Enter your email and password and verify your email"))
    items.append(bullet("Click <b>Billing</b> in the left sidebar"))
    items.append(bullet("Click <b>Add Payment Method</b> and enter your credit card"))
    items.append(bullet("Click <b>Add Credits</b> and add <b>$30</b> to start"))
    items.append(info_box("Why $30?", [
        "Full training run costs $10–20 at $1.89/hr",
        "Extra $10 covers setup time, smoke test, and any mistakes",
        "Unused credits never expire — safe to over-fund",
    ]))
    items.append(spacer(3))

    items.append(step_box("1.2", "Add Your SSH Key to RunPod", "~3 minutes"))
    items.append(spacer(2))
    items.append(body("Before launching a pod, add your SSH key so you can connect without a password."))
    items.append(bullet("Click <b>Settings</b> in the left sidebar (gear icon)"))
    items.append(bullet("Click <b>SSH Public Keys</b> → <b>Add SSH Key</b>"))
    items.append(bullet("Paste the entire contents of your public key (from step 0.3)"))
    items.append(bullet("Give it a name: <b>mac-paryaya</b> → click <b>Save</b>"))
    items.append(spacer(3))

    items.append(step_box("1.3", "Launch the A100 Pod", "~10 minutes"))
    items.append(spacer(2))
    items.append(warn_box(
        "Use Secure Cloud, not Community Cloud. Community Cloud pods can be interrupted "
        "mid-training with no warning. Secure Cloud pods are stable and guaranteed."))
    items.append(spacer(2))
    items.append(bullet("Click <b>Secure Cloud</b> in the left sidebar"))
    items.append(bullet("Find <b>A100 PCIe 40GB</b> (~$1.89/hr) in the GPU list"))
    items.append(body("If A100 PCIe 40GB is not available, use <b>A100 SXM 80GB</b> (~$2.79/hr). "
                      "It is faster but costs slightly more."))
    items.append(bullet("Click <b>Deploy</b> next to your chosen GPU"))
    items += h3("Pod Configuration — fill in these exact settings:")
    items.append(spacer(1))
    items.append(data_table(
        ["Setting", "Value", "Why"],
        [
            ["Template",           "RunPod PyTorch 2.2",  "Pre-installed CUDA 12.1 + Python — saves 30 min setup"],
            ["Container Disk",     "30 GB",               "For OS, conda, and pip packages (~15 GB needed)"],
            ["Volume Disk",        "50 GB",               "For FLEURS (~3 GB) + checkpoints + HF model cache (~5 GB)"],
            ["Volume Mount Path",  "/workspace",          "Where all data will live — must be /workspace"],
            ["Expose TCP Ports",   "22",                  "Required for SSH access"],
        ],
        col_widths=[TW*0.25, TW*0.22, TW*0.53],
    ))
    items.append(spacer(2))
    items.append(warn_box(
        "Volume Disk must be at least 50 GB. Google FLEURS downloads ~3 GB of audio. "
        "HuggingFace caches Whisper model weights (~3 GB) and tokenizers. "
        "Running out of disk mid-training corrupts your checkpoint and wastes GPU hours."))
    items.append(spacer(2))
    items.append(bullet("Click <b>Deploy</b> and wait <b>2–5 minutes</b> for the pod to reach <b>Running</b> status"))
    items.append(spacer(3))

    items.append(step_box("1.4", "Connect to Your Pod via SSH", "~3 minutes"))
    items.append(spacer(2))
    items.append(bullet("Once the pod shows <b>Running</b>, click <b>Connect</b>"))
    items.append(bullet("Click <b>SSH over exposed TCP</b> — you will see a command like:"))
    items.append(code_block("ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519"))
    items.append(bullet("Copy that exact command and run it in your Mac terminal"))
    items.append(body("If the connection works you will see:"))
    items.append(code_block("root@A5F8D3B2:~#"))
    items += h3("If Connection Fails")
    items.append(code_block(
        "# Load your key into the SSH agent first:\nssh-add ~/.ssh/id_ed25519\n\n"
        "# Then retry with StrictHostKeyChecking disabled:\n"
        "ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"))
    items.append(spacer(3))

    items.append(step_box("1.5", "Verify the GPU is Working", "~2 minutes"))
    items.append(spacer(2))
    items.append(body("Run these commands on the <b>RunPod terminal</b> (after SSH-ing in):"))
    items.append(code_block("nvidia-smi"))
    items.append(body("Expected: shows <b>A100-PCIE-40GB</b> with 40960 MiB memory."))
    items.append(code_block(
        "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available()); \"\\\n"
        "         \"print('GPU:', torch.cuda.get_device_name(0))\""))
    items.append(body("Expected output:"))
    items.append(code_block("CUDA: True\nGPU: NVIDIA A100-PCIE-40GB"))
    items.append(body("<b>If CUDA shows False</b>, reinstall PyTorch with CUDA support:"))
    items.append(code_block(
        "pip uninstall torch torchaudio -y\n"
        "pip install torch==2.2.0 torchaudio==2.2.0 \\\n"
        "    --index-url https://download.pytorch.org/whl/cu121"))
    items.append(PageBreak())
    return items


def build_part2() -> list:
    items = []
    items += h1("PART 2 — SET UP THE POD")
    items.append(Paragraph("~15 minutes — runs mostly automatically", S("_tm", fontSize=9,
                            leading=13, fontName="Helvetica-Oblique", textColor=C_MUTED, spaceAfter=8)))

    items.append(step_box("2.1", "Move HuggingFace Cache to the Persistent Volume", "~1 minute"))
    items.append(spacer(2))
    items.append(warn_box(
        "By default HuggingFace downloads go to ~/.cache/ on the container disk, which is "
        "wiped on pod restart. This command moves the cache to /workspace (the persistent "
        "volume) so downloads survive disconnection. Run it before anything else."))
    items.append(spacer(2))
    items.append(code_block(
        "echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc\n"
        "echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets' >> ~/.bashrc\n"
        "source ~/.bashrc\n"
        "mkdir -p /workspace/.cache/huggingface/datasets"))
    items.append(spacer(3))

    items.append(step_box("2.2", "Clone the Paryaya Repository", "~2 minutes"))
    items.append(spacer(2))
    items.append(code_block(
        "cd /workspace\ngit clone https://github.com/asticrat/paryaya.git\ncd paryaya"))
    items.append(body("Verify the key files are present:"))
    items.append(code_block(
        "ls scripts/\n# Should include: finetune_whisper.py  setup_runpod_whisper.sh\n\n"
        "ls configs/\n# Should include: finetune_whisper.yaml"))
    items.append(spacer(3))

    items.append(step_box("2.3", "Run the Setup Script", "~10 minutes"))
    items.append(spacer(2))
    items.append(code_block("bash scripts/setup_runpod_whisper.sh"))
    items.append(body("The script installs: git, ffmpeg, libsndfile1, tmux, transformers, "
                      "datasets, accelerate, evaluate, jiwer, librosa, soundfile, wandb, pyyaml."))
    items.append(body("Expected ending output:"))
    items.append(code_block(
        "✅ System deps installed\n"
        "✅ Repo ready at /workspace/paryaya\n"
        "✅ Python deps installed\n"
        "CUDA available : True\n"
        "GPU            : NVIDIA A100-PCIE-40GB\n"
        "VRAM           : 40.1 GB"))
    items.append(body("If a package fails to install, run manually:"))
    items.append(code_block(
        "pip install \"transformers>=4.43\" \"datasets>=2.21\" \"accelerate>=0.33\" \\\n"
        "    \"evaluate>=0.4\" \"jiwer>=3.0\" soundfile librosa wandb pyyaml \\\n"
        "    \"huggingface-hub>=0.24\"\npip install -e ."))
    items.append(spacer(3))

    items.append(step_box("2.4", "Set W&B Key (Optional)", "~1 minute"))
    items.append(spacer(2))
    items.append(tip_box(
        "HF_TOKEN is NOT required. Google FLEURS is fully public and downloads without authentication. "
        "Only set WANDB_API_KEY if you want real-time charts in your browser."))
    items.append(spacer(2))
    items.append(code_block(
        "# WandB key (optional — for real-time training charts):\nexport WANDB_API_KEY=abcdef1234567890abcdef1234567890abcdef12\n"
        "export WANDB_PROJECT=paryaya-whisper\n\n"
        "# Save so it persists if pod restarts:\n"
        "echo \"export WANDB_API_KEY=$WANDB_API_KEY\" >> ~/.bashrc\n"
        "echo \"export WANDB_PROJECT=paryaya-whisper\" >> ~/.bashrc\n\n"
        "# Verify it was set:\n"
        "echo \"WANDB: $([ -n \\\"$WANDB_API_KEY\\\" ] && echo SET || echo not set)\"\n"
        "# If not using W&B, that's fine — training works without it"))
    items.append(spacer(3))

    items.append(step_box("2.5", "Log In to W&B", "~2 minutes (skip if not using W&B)"))
    items.append(spacer(2))
    items.append(code_block(
        "wandb login\n# Paste your API key when prompted, then press Enter\n\n"
        "wandb status\n# Should show: Currently logged in as: YOUR_USERNAME"))
    items.append(body("If you see <i>wandb: ERROR</i>, your key is wrong. "
                      "Get a new one at wandb.ai/settings → API Keys."))
    items.append(spacer(3))

    items.append(step_box("2.6", "Final Readiness Check", "~2 minutes"))
    items.append(spacer(2))
    items.append(body("Run this block as-is. All lines must show ✅ before training:"))
    items.append(code_block(
        "python3 - <<'EOF'\nimport sys, os, shutil\nerrors = []\n\nimport torch\n"
        "if not torch.cuda.is_available():\n    errors.append(\"CUDA not available\")\n"
        "else:\n    gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
        "    print(f\"✅  GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)\")\n\n"
        "try:\n    import transformers; print(f\"✅  transformers: {transformers.__version__}\")\n"
        "except: errors.append(\"transformers not installed\")\n\n"
        "try:\n    import datasets; print(f\"✅  datasets: {datasets.__version__}\")\n"
        "except: errors.append(\"datasets not installed\")\n\n"
        "# HF_TOKEN not required for FLEURS — just note if set\n"
        "token = os.getenv(\"HF_TOKEN\", \"\")\n"
        "if token.startswith(\"hf_\"): print(f\"ℹ️   HF_TOKEN: set (optional, not needed for FLEURS)\")\n"
        "else: print(\"ℹ️   HF_TOKEN: not set (OK — FLEURS is public)\")\n\n"
        "free_gb = shutil.disk_usage(\"/workspace\").free / 1e9\n"
        "if free_gb < 15: errors.append(f\"Low disk: {free_gb:.1f} GB free (need 15+)\")\n"
        "else: print(f\"✅  Disk: {free_gb:.1f} GB free\")\n\n"
        "if errors:\n    print(\"\\n❌  FIX THESE:\")\n"
        "    for e in errors: print(f\"   - {e}\")\n"
        "    sys.exit(1)\nelse: print(\"\\n🟢  All checks passed!\")\nEOF"))
    items.append(PageBreak())
    return items


def build_part3() -> list:
    items = []
    items += h1("PART 3 — SMOKE TEST")
    items.append(Paragraph("~5 minutes — do not skip this", S("_tm", fontSize=9,
                            leading=13, fontName="Helvetica-Oblique", textColor=C_MUTED, spaceAfter=8)))

    items.append(warn_box(
        "The smoke test runs the full training pipeline on only 8 samples and 2 optimizer steps. "
        "It downloads a tiny slice of FLEURS, preprocesses it, and confirms the model "
        "trains and saves. Finding a bug here costs 5 minutes. Finding it after 3 hours of GPU "
        "time costs money and time that cannot be recovered."))
    items.append(spacer(3))

    items.append(step_box("3.1", "Run the Smoke Test", "~5 minutes"))
    items.append(spacer(2))
    items.append(code_block(
        "cd /workspace/paryaya\n\n"
        "python3 scripts/finetune_whisper.py \\\n"
        "    --config configs/finetune_whisper.yaml \\\n"
        "    --smoke_test"))
    items.append(spacer(2))
    items += h3("Expected Output (Normal)")
    items.append(code_block(
        "Loading processor from openai/whisper-medium ...\n"
        "Downloading model.safetensors: 100%|████████| 1.52G/1.52G ...\n\n"
        "Loading google/fleurs (ne_np) ...\n"
        "Downloading data: 100%|████████| ...\n\n"
        "  ⚡ Smoke test mode — 2 steps only\n"
        "Pre-processing audio + tokenising transcripts ...\n"
        "  train=8  eval=4\n\n"
        "Loading openai/whisper-medium weights ...\n\n"
        "🚀 Training on cuda | steps=2 | fp16=True\n"
        "   Checkpoint dir: checkpoints/whisper-medium-nepali\n\n"
        "{'loss': 8.2341, 'learning_rate': ..., 'epoch': ...}\n"
        "{'eval_loss': ..., 'eval_wer': ..., ...}\n\n"
        "✅ Training complete.\n"
        "   Best model saved → checkpoints/whisper-medium-nepali/best"))
    items.append(spacer(2))
    items.append(tip_box("The smoke test passes if you see '🚀 Training on cuda' "
                         "and '✅ Training complete.' with no Python traceback."))
    items.append(spacer(3))

    items += h3("Smoke Test Error Reference")
    items.append(data_table(
        ["Error", "Cause", "Fix"],
        [
            ["CUDA out of memory",
             "Batch size too large for this GPU",
             "Edit config: per_device_train_batch_size: 8  eval_batch_size: 4"],
            ["No module named 'evaluate'",
             "evaluate package not installed",
             "pip install evaluate>=0.4 jiwer"],
            ["No space left on device",
             "Volume disk too small",
             "Need at least 15 GB free on /workspace — recreate pod with 50 GB"],
            ["ConnectionError / Timeout loading FLEURS",
             "Network blip on RunPod",
             "Rerun the same command — HF datasets auto-resumes partial downloads"],
            ["KeyError: 'transcription'",
             "Wrong text_column in config",
             "Verify configs/finetune_whisper.yaml has text_column: transcription"],
        ],
        col_widths=[TW*0.30, TW*0.30, TW*0.40],
    ))
    items.append(body("Fix any error and rerun the smoke test before proceeding to full training."))
    items.append(PageBreak())
    return items


def build_part4() -> list:
    items = []
    items += h1("PART 4 — FULL TRAINING")
    items.append(Paragraph("~3–6 hours unattended — start before bed or when you can leave it running",
                            S("_tm", fontSize=9, leading=13, fontName="Helvetica-Oblique",
                              textColor=C_MUTED, spaceAfter=8)))

    items.append(step_box("4.1", "Start a tmux Session", "~1 minute"))
    items.append(spacer(2))
    items.append(body(
        "tmux keeps training running even after you close your SSH connection. "
        "Without it, disconnecting your terminal kills training."))
    items.append(code_block("# Create a persistent session named 'train':\ntmux new -s train"))
    items.append(body("You are now inside tmux. The green bar at the bottom of your terminal "
                      "confirms you are in the session."))
    items.append(spacer(3))

    items.append(step_box("4.2", "Launch Full Training", "~3–6 hours"))
    items.append(spacer(2))
    items.append(code_block(
        "cd /workspace/paryaya\n\n"
        "python3 scripts/finetune_whisper.py \\\n"
        "    --config configs/finetune_whisper.yaml"))
    items.append(body("Training first downloads Google FLEURS Nepali (first run only, ~3 GB, "
                      "3–5 minutes on RunPod), then preprocesses the audio, and begins training."))
    items.append(spacer(2))
    items += h3("What You Will See")
    items.append(code_block(
        "Loading processor from openai/whisper-medium ...\n"
        "Loading google/fleurs (ne_np) ...\n"
        "Pre-processing audio + tokenising transcripts ...\n"
        "  train=3,081  eval=579\n\n"
        "Loading openai/whisper-medium weights ...\n\n"
        "🚀 Training on cuda | steps=3000 | fp16=True\n"
        "   Checkpoint dir: checkpoints/whisper-medium-nepali\n\n"
        "{'loss': 8.4123, 'learning_rate': 2e-08, 'epoch': 0.09}\n"
        "{'loss': 7.8234, 'learning_rate': 4e-07, 'epoch': 0.18}\n"
        "...improving slowly over hours..."))
    items.append(spacer(2))

    items += h3("Training Progress — What to Expect")
    items.append(data_table(
        ["Steps", "Loss", "WER", "What is Happening"],
        [
            ["1–100",    "7–9",   "80–95%", "Model aligning to Devanagari output — looks terrible, completely normal"],
            ["100–500",  "5–7",   "50–80%", "Learning Nepali phoneme patterns — WER drops fast"],
            ["500–2000", "2–5",   "20–50%", "Real words forming, model gains vocabulary"],
            ["2000–3500","1.5–3", "15–30%", "Fine-tuning word boundaries and context"],
            ["2000–3000","0.8–2", "8–22%",  "Final polish — target is WER below 20%"],
        ],
        col_widths=[TW*0.12, TW*0.12, TW*0.12, TW*0.64],
    ))
    items.append(spacer(2))
    items.append(info_box("Do Not Panic at High Early WER", [
        "WER of 80–95% in the first 100 steps is completely normal",
        "Whisper already understands Nepali from pre-training — fine-tuning just aligns it",
        "Loss dropping from 8 to 7 in the first 50 steps confirms training is working",
        "Only worry if loss is still above 7.5 at step 300+",
    ]))
    items.append(spacer(3))

    items.append(step_box("4.3", "Detach from tmux and Leave Training Running", "~30 seconds"))
    items.append(spacer(2))
    items.append(body("Once training is running and you see loss values printing, detach:"))
    items.append(code_block("# Press Ctrl+B, then D  (hold Ctrl, press B, release both, then press D)"))
    items.append(body("You will see:"))
    items.append(code_block("[detached (from session train)]"))
    items.append(body("Training continues on the pod. You can now safely close your terminal, "
                      "turn off your laptop, or go to sleep."))
    items.append(spacer(3))

    items.append(step_box("4.4", "Monitor Training Progress", "Ongoing"))
    items.append(spacer(2))
    items += h3("Option A — W&B Dashboard (recommended)")
    items.append(body("Go to <b>wandb.ai/YOUR_USERNAME/paryaya-whisper</b> in your browser. "
                      "You will see live charts for:"))
    items.append(bullet("<b>train/loss</b> — should decrease steadily throughout training"))
    items.append(bullet("<b>eval/wer</b> — the key metric, target below 0.20 (20%) by end"))
    items.append(bullet("<b>eval/loss</b> — decreases with small bumps, that is normal"))
    items.append(spacer(2))
    items += h3("Option B — From the Terminal")
    items.append(code_block(
        "# Reconnect to pod and reattach:\nssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519\ntmux attach -t train\n\n"
        "# In a second SSH connection — check GPU is being used (should be 85-99%):\nwatch -n 5 nvidia-smi\n\n"
        "# Check checkpoints are being saved:\nls -lht /workspace/paryaya/checkpoints/whisper-medium-nepali/\n"
        "# Expect: checkpoint-500  checkpoint-1000  checkpoint-1500  ... updating regularly"))
    items.append(spacer(3))

    items.append(step_box("4.5", "Reconnecting After Your SSH Drops", "~2 minutes"))
    items.append(spacer(2))
    items.append(body("Training is still running (tmux keeps it alive). Simply reconnect:"))
    items.append(code_block(
        "ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519\ntmux attach -t train"))
    items += h3("If the Pod Restarted (tmux session gone)")
    items.append(body("Check if checkpoints were saved to the persistent volume:"))
    items.append(code_block(
        "ls /workspace/paryaya/checkpoints/whisper-medium-nepali/\n"
        "# If checkpoint folders exist, resume from them:"))
    items.append(code_block(
        "source ~/.bashrc          # reload WANDB keys\ntmux new -s train\n"
        "cd /workspace/paryaya\n\n"
        "# The trainer automatically detects the latest checkpoint\n"
        "# in the output_dir and resumes from it:\n"
        "python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml"))
    items.append(spacer(3))

    items.append(step_box("4.6", "Early Stopping Behaviour", "Automatic"))
    items.append(spacer(2))
    items.append(body(
        "The config includes <b>early_stopping_patience: 5</b>. If validation WER does not "
        "improve for 5 consecutive evaluations (5 × 500 steps = 2500 steps without improvement), "
        "training stops automatically and saves the best checkpoint."))
    items.append(tip_box("If training stops before step 4000, that is fine. "
                         "Early stopping found the optimal point. The best model is saved."))
    items.append(spacer(3))

    items.append(step_box("4.7", "When Training Finishes", "~2 minutes"))
    items.append(spacer(2))
    items.append(body("You will see the final output:"))
    items.append(code_block(
        "✅ Training complete.\n"
        "   Best model saved → checkpoints/whisper-medium-nepali/best\n\n"
        "   Deploy to Paryaya API:\n"
        "   export ASR_BACKEND=whisper\n"
        "   export WHISPER_MODEL_PATH=checkpoints/whisper-medium-nepali/best\n"
        "   uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000"))
    items.append(body("Check the best WER achieved during training:"))
    items.append(code_block(
        "cat /workspace/paryaya/checkpoints/whisper-medium-nepali/trainer_state.json \\\n"
        "  | python3 -c \"\nimport sys, json\ns = json.load(sys.stdin)\n"
        "best = min(s['log_history'], key=lambda x: x.get('eval_wer', 999))\n"
        "print(f\\\"Best WER: {best.get('eval_wer', 'N/A'):.1%} at step {best.get('step','?')}\\\")\n\""))
    items.append(info_box("Interpreting Your WER", [
        "WER below 20%  — Excellent result, model is production-ready",
        "WER 20–35%     — Good result, noticeably better than baseline Whisper on Nepali",
        "WER 35–50%     — Partial success — consider more training steps or data",
        "WER above 50%  — Something went wrong, check that dataset was loaded correctly",
    ]))
    items.append(PageBreak())
    return items


def build_part5() -> list:
    items = []
    items += h1("PART 5 — DOWNLOAD MODEL AND TERMINATE POD")
    items.append(Paragraph("~15 minutes — do not skip any step",
                            S("_tm", fontSize=9, leading=13, fontName="Helvetica-Oblique",
                              textColor=C_MUTED, spaceAfter=8)))

    items.append(warn_box(
        "Once you terminate the pod, the volume data is permanently deleted. "
        "Download everything listed below before clicking Terminate. "
        "Double-check each file downloaded successfully before terminating."))
    items.append(spacer(3))

    items.append(step_box("5.1", "Verify the Model Was Saved on the Pod", "~2 minutes"))
    items.append(spacer(2))
    items.append(body("Run on the RunPod terminal:"))
    items.append(code_block(
        "ls -lh /workspace/paryaya/checkpoints/whisper-medium-nepali/best/"))
    items.append(body("Expected — these files must all be present:"))
    items.append(code_block(
        "config.json\ngeneration_config.json\nmodel.safetensors       (~1.5 GB)\n"
        "preprocessor_config.json\nspecial_tokens_map.json\ntokenizer.json\n"
        "tokenizer_config.json\nvocab.json\nadded_tokens.json\nmerges.txt"))
    items.append(body("If the <b>best/</b> folder is empty or missing, save manually:"))
    items.append(code_block(
        "python3 - <<'EOF'\nimport json\nfrom pathlib import Path\n"
        "from transformers import WhisperForConditionalGeneration, WhisperProcessor\n\n"
        "state = json.loads(Path(\"checkpoints/whisper-medium-nepali/trainer_state.json\").read_text())\n"
        "best = state.get(\"best_model_checkpoint\", \"checkpoints/whisper-medium-nepali/checkpoint-500\")\n"
        "print(f\"Saving from: {best}\")\n\n"
        "model = WhisperForConditionalGeneration.from_pretrained(best)\n"
        "proc  = WhisperProcessor.from_pretrained(best)\n"
        "Path(\"checkpoints/whisper-medium-nepali/best\").mkdir(parents=True, exist_ok=True)\n"
        "model.save_pretrained(\"checkpoints/whisper-medium-nepali/best\")\n"
        "proc.save_pretrained(\"checkpoints/whisper-medium-nepali/best\")\n"
        "print(\"Done\")\nEOF"))
    items.append(spacer(3))

    items.append(step_box("5.2", "Download the Model to Your Mac", "~5–10 minutes"))
    items.append(spacer(2))
    items.append(body("Open a <b>new terminal on your Mac</b> (leave the RunPod SSH session open):"))
    items.append(code_block(
        "# Create a folder to receive the model:\nmkdir -p ~/paryaya_model\n\n"
        "# Download the entire best/ folder (~1.5 GB, takes 5-10 minutes)\n"
        "# Replace XX.XX.XX.XX and XXXXX with your actual pod IP and port:\n"
        "scp -P XXXXX -r \\\n"
        "    root@XX.XX.XX.XX:/workspace/paryaya/checkpoints/whisper-medium-nepali/best/ \\\n"
        "    ~/paryaya_model/whisper-medium-nepali/"))
    items.append(body("Verify the download:"))
    items.append(code_block(
        "ls -lh ~/paryaya_model/whisper-medium-nepali/\n"
        "# model.safetensors should be ~1.5 GB\n\n"
        "du -sh ~/paryaya_model/\n# Should show ~1.5–1.6 GB total"))
    items.append(body("Quick load test to confirm the file is not corrupted:"))
    items.append(code_block(
        "cd ~/paryaya_model\npython3 - <<'EOF'\n"
        "from transformers import WhisperForConditionalGeneration, WhisperProcessor\n"
        "m = WhisperForConditionalGeneration.from_pretrained(\"whisper-medium-nepali\")\n"
        "print(f\"✅ Loaded: {m.num_parameters():,} params | lang: {m.generation_config.language}\")\n"
        "EOF\n# Expected: ✅ Loaded: 307,198,976 params | lang: nepali"))
    items.append(spacer(3))

    items.append(step_box("5.3", "Copy the Model into Your Paryaya Project", "~1 minute"))
    items.append(spacer(2))
    items.append(code_block(
        "cp -r ~/paryaya_model/whisper-medium-nepali/ \\\n"
        "    /Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali/\n\n"
        "echo \"Model is now at:\"\n"
        "ls /Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali/"))
    items.append(spacer(3))

    items.append(step_box("5.4", "Terminate the RunPod Pod", "~3 minutes"))
    items.append(spacer(2))
    items.append(body("Only do this after verifying the download in steps 5.2 and 5.3."))
    items.append(bullet("Go to <b>runpod.io → My Pods</b>"))
    items.append(bullet("Find your pod — it shows as <b>Running</b>"))
    items.append(bullet("Click the <b>three dots (...)</b> menu on the right side of your pod"))
    items.append(bullet("Click <b>Terminate Pod</b>"))
    items.append(bullet("Confirm by clicking <b>Terminate</b> in the dialog"))
    items.append(bullet("The pod disappears from the list and billing stops immediately"))
    items.append(spacer(2))
    items.append(warn_box(
        "Do NOT just stop the pod (the Stop button). A stopped pod still charges for "
        "volume storage. Use Terminate to stop all billing immediately."))
    items.append(spacer(2))
    items.append(body("After terminating, confirm your total spend: "
                      "<b>runpod.io → Billing → Usage</b>. Should be under $20."))
    items.append(PageBreak())
    return items


def build_part6() -> list:
    items = []
    items += h1("PART 6 — DEPLOY VIA PARYAYA API")
    items.append(Paragraph("~10 minutes", S("_tm", fontSize=9, leading=13,
                            fontName="Helvetica-Oblique", textColor=C_MUTED, spaceAfter=8)))

    items.append(step_box("6.1", "Test Locally First", "~5 minutes"))
    items.append(spacer(2))
    items.append(code_block(
        "cd /Users/yaxzyra/Documents/asti-lab/paryaya\nsource .venv/bin/activate\n\n"
        "export ASR_BACKEND=whisper\n"
        "export WHISPER_MODEL_PATH=/Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali\n\n"
        "uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000"))
    items.append(body("Expected startup logs:"))
    items.append(code_block(
        "INFO  Starting Paryaya API | backend=whisper device=mps\n"
        "INFO  Whisper backend loaded: .../checkpoints/whisper-medium-nepali\n"
        "INFO  Application startup complete."))
    items.append(body("In a new terminal, test the API:"))
    items.append(code_block(
        "curl http://localhost:8000/health\n"
        "# Expected: {\"status\":\"ok\",\"model_loaded\":true,\"backend\":\"whisper\",...}\n\n"
        "curl -X POST http://localhost:8000/v1/transcribe \\\n"
        "    -H \"Authorization: Bearer sk-paryaya-testkey123\" \\\n"
        "    -F \"file=@/path/to/nepali_audio.wav\""))
    items.append(spacer(3))

    items.append(step_box("6.2", "Docker Deployment", "~5 minutes"))
    items.append(spacer(2))
    items.append(body("Add these environment variables to your <b>.env</b> file or "
                      "docker-compose.yml environment section:"))
    items.append(code_block(
        "ASR_BACKEND=whisper\n"
        "WHISPER_MODEL_PATH=/app/checkpoints/whisper-medium-nepali"))
    items.append(body("In <b>docker/docker-compose.yml</b>, add a volume mount if not already present:"))
    items.append(code_block(
        "services:\n  api:\n    environment:\n"
        "      - ASR_BACKEND=whisper\n"
        "      - WHISPER_MODEL_PATH=/app/checkpoints/whisper-medium-nepali\n"
        "    volumes:\n      - ./checkpoints:/app/checkpoints"))
    items.append(body("Deploy:"))
    items.append(code_block(
        "docker compose -f docker/docker-compose.yml up -d\n"
        "docker compose -f docker/docker-compose.yml logs -f api"))
    items.append(PageBreak())
    return items


def build_troubleshooting() -> list:
    items = []
    items += h1("TROUBLESHOOTING")

    items += h2("Training Problems")

    items += h3("Loss stuck above 7.5 after step 200")
    items.append(body("Steps 1–100 are always slow. If still stuck at step 200+, check:"))
    items.append(code_block(
        "# Verify fp16 is actually being used:\ngrep 'fp16' configs/finetune_whisper.yaml\n# Should show: fp16: true\n\n"
        "# Check GPU memory is being used (should be 25–38 GB):\nnvidia-smi\n# If under 10 GB, fp16 is falling back to fp32"))

    items += h3("CUDA out of memory")
    items.append(code_block(
        "# Reduce batch size in configs/finetune_whisper.yaml:\n"
        "# Change: per_device_train_batch_size: 16  →  8\n"
        "# Change: per_device_eval_batch_size: 8   →  4\n"
        "# Change: gradient_accumulation_steps: 2  →  4  (keeps effective batch = 32)\n"
        "# Save and rerun training"))

    items += h3("W&B not logging / wandb: ERROR")
    items.append(code_block(
        "wandb login --relogin   # re-enter your API key\n\n"
        "# Or disable W&B entirely:\nexport WANDB_MODE=disabled\npython3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml"))

    items += h3("evaluate.load('wer') fails")
    items.append(code_block(
        "pip install evaluate>=0.4 jiwer>=3.0\npython3 -c \"import evaluate; m = evaluate.load('wer'); print('OK')\""))

    items += h3("Training is very slow (>5 seconds per step)")
    items.append(body(
        "The first run includes audio preprocessing via the map() function which is slow. "
        "Once preprocessing finishes and actual training steps begin, each step should take "
        "0.3–0.8 seconds. If steps are still slow after 30 minutes, check GPU utilisation "
        "with nvidia-smi."))

    items += h2("Dataset Problems")

    items += h3("ConnectionError or timeout when downloading FLEURS")
    items.append(body("Google FLEURS downloads from HuggingFace CDN. If the download fails partway:"))
    items.append(code_block(
        "# Just rerun the same training command — HuggingFace datasets\n"
        "# automatically resumes partial downloads from where they stopped.\n"
        "python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml --smoke_test"))

    items += h3("KeyError: 'transcription'")
    items.append(body("The text column name in the config does not match the dataset. Verify:"))
    items.append(code_block(
        "grep text_column configs/finetune_whisper.yaml\n"
        "# Should show: text_column: \"transcription\"\n\n"
        "# If missing, add it:\necho '  text_column: \"transcription\"' >> configs/finetune_whisper.yaml"))

    items += h3("Dataset download is slow")
    items.append(body("FLEURS Nepali is ~3 GB and should download in 3–8 minutes on RunPod. "
                      "If slower than 15 minutes, there may be a RunPod network issue — "
                      "wait it out rather than restarting."))

    items += h2("Connection Problems")

    items += h3("SSH disconnects frequently")
    items.append(code_block(
        "# Add keepalive flags to your SSH command:\nssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519 \\\n"
        "    -o ServerAliveInterval=60 \\\n    -o ServerAliveCountMax=10"))

    items += h3("GitHub SSH: connection timed out on port 22")
    items.append(code_block(
        "# Add this to ~/.ssh/config on your Mac:\nnano ~/.ssh/config\n\n"
        "# Add these lines:\nHost github.com\n  Hostname ssh.github.com\n  Port 443\n"
        "  AddKeysToAgent yes\n  IdentityFile ~/.ssh/id_ed25519"))

    items += h3("Pod IP/port changed after reconnect")
    items.append(body(
        "Pod IPs and ports can change after a restart. Always get the current SSH command "
        "from <b>runpod.io → My Pods → Connect</b> rather than using a saved command."))

    items += h2("API Problems")

    items += h3("API returns 503 model_loaded: false")
    items.append(code_block(
        "# Check that WHISPER_MODEL_PATH points to a valid directory:\nls $WHISPER_MODEL_PATH\n"
        "# Should show model.safetensors and config.json\n\n"
        "# Check the API startup logs:\ndocker compose -f docker/docker-compose.yml logs api | tail -20"))

    items += h3("Latency is high (>3 seconds per request)")
    items.append(body(
        "On CPU (MPS on Mac), Whisper medium takes 2–4 seconds per clip. On a server "
        "without GPU, this is expected. On GPU it should be 200–400ms. "
        "Set BEAM_WIDTH=1 in your environment for faster (but slightly less accurate) decoding."))

    items.append(PageBreak())
    return items


def build_reference() -> list:
    items = []
    items += h1("QUICK REFERENCE AND COST SUMMARY")

    items += h2("Commands You Will Use Most")
    items.append(data_table(
        ["Action", "Command"],
        [
            ["Connect to pod",         "ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519"],
            ["Reattach to training",   "tmux attach -t train"],
            ["Detach from tmux",       "Ctrl+B, then D"],
            ["Check GPU",              "nvidia-smi  (in second SSH session)"],
            ["Check checkpoints",      "ls -lht /workspace/paryaya/checkpoints/whisper-medium-nepali/"],
            ["Check disk",             "df -h /workspace"],
            ["Reload env vars",        "source ~/.bashrc"],
            ["Run smoke test",         "python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml --smoke_test"],
            ["Download model (Mac)",   "scp -P XXXXX -r root@XX.XX.XX.XX:/workspace/paryaya/checkpoints/whisper-medium-nepali/best/ ~/paryaya_model/whisper-medium-nepali/"],
            ["Start API (local)",      "export ASR_BACKEND=whisper && uvicorn paryaya.api.main:app --port 8000"],
            ["Check API health",       "curl http://localhost:8000/health"],
        ],
        col_widths=[TW*0.28, TW*0.72],
    ))
    items.append(spacer(4))

    items += h2("Cost Summary")
    items.append(data_table(
        ["Phase", "Duration", "Cost at $1.89/hr"],
        [
            ["Account setup (no pod running)", "30 min",  "Free"],
            ["Pod setup + smoke test",          "30 min",  "~$0.95"],
            ["Dataset download + preprocessing","30 min",  "~$0.95"],
            ["Full training (3000 steps, FLEURS)","2–3 hrs", "~$3.78–5.67"],
            ["Download model + terminate",      "20 min",  "~$0.63"],
            ["Total",                           "4.5–7 hrs","~$8–12"],
        ],
        col_widths=[TW*0.40, TW*0.25, TW*0.35],
    ))
    items.append(spacer(2))
    items.append(info_box("Budget Check", [
        "A100 PCIe 40GB runs at ~$1.89/hr on RunPod Secure Cloud (price may vary)",
        "A6-hour full run (including setup) costs ~$11.34",
        "With $30 credit loaded, you have 4x more than needed — no risk of running out",
        "Unused RunPod credits never expire",
    ]))
    items.append(spacer(4))

    items += h2("Environment Variables for Deployment")
    items.append(data_table(
        ["Variable", "Value", "Required?"],
        [
            ["ASR_BACKEND",         "whisper",                               "Yes"],
            ["WHISPER_MODEL_PATH",  "/path/to/checkpoints/whisper-medium-nepali", "Yes"],
            ["MAX_AUDIO_MB",        "50 (default)",                         "No"],
            ["BEAM_WIDTH",          "5 (lower = faster, slightly less accurate)", "No"],
            ["LOG_LEVEL",           "INFO",                                 "No"],
        ],
        col_widths=[TW*0.32, TW*0.48, TW*0.20],
    ))
    items.append(spacer(4))

    items += h2("File Locations After Completion")
    items.append(data_table(
        ["File/Folder", "Location"],
        [
            ["Fine-tuned model",       "/Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali/"],
            ["Training config",        "/Users/yaxzyra/Documents/asti-lab/paryaya/configs/finetune_whisper.yaml"],
            ["Fine-tuning script",     "/Users/yaxzyra/Documents/asti-lab/paryaya/scripts/finetune_whisper.py"],
            ["RunPod setup script",    "/Users/yaxzyra/Documents/asti-lab/paryaya/scripts/setup_runpod_whisper.sh"],
            ["Whisper API backend",    "/Users/yaxzyra/Documents/asti-lab/paryaya/src/paryaya/inference/whisper_backend.py"],
            ["This guide (markdown)",  "/Users/yaxzyra/Documents/asti-lab/paryaya/docs/runpod_finetune_guide.md"],
        ],
        col_widths=[TW*0.30, TW*0.70],
    ))
    items.append(spacer(6))
    items.append(rule())
    items.append(spacer(2))
    items.append(Paragraph(
        "Paryaya — पर्याय — many voices, one understanding.",
        S("_fin", fontSize=10, leading=14, fontName="Helvetica-Oblique",
          textColor=C_MUTED, alignment=TA_CENTER)))
    items.append(Paragraph(
        "github.com/asticrat/paryaya",
        S("_gh", fontSize=9, leading=13, fontName="Helvetica",
          textColor=C_BLUE, alignment=TA_CENTER)))
    return items


# ── Build document ────────────────────────────────────────────────────────────
def main():
    out_path = Path(__file__).parent / "Paryaya_RunPod_Guide.pdf"

    # Two page templates: cover (no header/footer) and normal
    cover_frame  = Frame(0, 0, W, H, 0, 0, 0, 0, id="cover")
    normal_frame = Frame(MARGIN_L, MARGIN_B, TW, H - MARGIN_T - MARGIN_B,
                         id="normal")

    cover_tpl  = PageTemplate("cover",  frames=[cover_frame],
                               onPage=_cover_page)
    normal_tpl = PageTemplate("normal", frames=[normal_frame],
                               onPage=_header_footer)

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        pageTemplates=[cover_tpl, normal_tpl],
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T,  bottomMargin=MARGIN_B,
        title="Paryaya — RunPod Whisper Fine-Tuning Guide",
        author="Paryaya / asticrat",
        subject="Nepali ASR fine-tuning on RunPod A100",
    )

    story = []

    # Cover (uses cover template)
    story += build_cover()

    # Switch to normal template for all remaining pages
    story.append(NextPageTemplate("normal"))

    # TOC
    story += build_toc()

    # Parts
    story += build_part0()
    story += build_part1()
    story += build_part2()
    story += build_part3()
    story += build_part4()
    story += build_part5()
    story += build_part6()
    story += build_troubleshooting()
    story += build_reference()

    doc.build(story)
    print(f"✅  PDF generated: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
