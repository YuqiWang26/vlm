"""Benchmark runner for visual-token compression experiments."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from src.compression import create_compression_method
from src.metrics import compute_quality_score, summarize_results
from src.model_loader import VLMEngine
from src.utils import ensure_dir, get_peak_gpu_memory_mb, reset_peak_gpu_memory


@dataclass
class ToySample:
    sample_id: str
    image: Image.Image
    question: str
    reference_answer: str
    keywords: List[str]


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    """Load a readable font on both Colab Linux and local macOS, with fallback."""

    names = []
    if mono:
        names.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/System/Library/Fonts/Menlo.ttc",
            ]
        )
    if bold:
        names.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            ]
        )
    names.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - bbox[1]
    draw.text((x, y), text, fill=fill, font=font)


def create_toy_dataset(image_size: int = 1024) -> List[ToySample]:
    """Create a synthetic stress VQA/OCR benchmark for Colab.

    These samples are intentionally harder than simple color/shape cards:
    small OCR, tiny tables, counting, spatial references, and multi-field answers.
    Aggressive visual-token compression is much more likely to lose one of the
    answer-critical details, so the benchmark can show an accuracy trade-off
    without depending on a large external dataset download.
    """

    samples: List[ToySample] = []
    title_font = _load_font(36, bold=True)
    text_font = _load_font(30)
    small_font = _load_font(24)
    code_font = _load_font(44, bold=True, mono=True)
    tiny_code_font = _load_font(34, bold=True, mono=True)

    # 1) Small OCR in corners. No-compression should usually read both; low
    # retention or proxy image resizing often drops at least one exact code.
    img = Image.new("RGB", (image_size, image_size), (250, 250, 247))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 944, 944], outline=(50, 50, 50), width=6)
    draw.text((115, 110), "Inspection card", fill=(35, 35, 35), font=title_font)
    draw.text((115, 180), "Most printed text is intentionally irrelevant.", fill=(70, 70, 70), font=text_font)
    draw.text((115, 230), "Read the two small colored labels near the bottom.", fill=(70, 70, 70), font=text_font)
    blue_box = (105, 810, 315, 900)
    yellow_box = (710, 810, 920, 900)
    draw.rounded_rectangle(blue_box, radius=12, fill=(178, 220, 255), outline=(60, 60, 60), width=4)
    draw.rounded_rectangle(yellow_box, radius=12, fill=(255, 245, 170), outline=(60, 60, 60), width=4)
    _draw_centered_text(draw, blue_box, "M2Q8", code_font)
    _draw_centered_text(draw, yellow_box, "K7P4", code_font)
    samples.append(
        ToySample(
            "small_corner_code",
            img,
            "Read the codes in the blue label and the yellow label. Answer with both codes only.",
            "M2Q8 K7P4",
            ["m2q8", "k7p4"],
        )
    )

    # 2) Exact value from a mini receipt. This is sensitive to small text.
    img = Image.new("RGB", (image_size, image_size), (242, 246, 250))
    draw = ImageDraw.Draw(img)
    receipt = (250, 90, 650, 805)
    draw.rectangle(receipt, fill=(255, 255, 255), outline=(35, 35, 35), width=4)
    draw.text((302, 125), "CITY CAFE", fill=(20, 20, 20), font=title_font)
    y = 205
    for item, price in [("TEA", "3.25"), ("SOUP", "6.50"), ("BREAD", "2.00"), ("TAX", "1.35")]:
        draw.text((290, y), item, fill=(40, 40, 40), font=small_font)
        draw.text((520, y), price, fill=(40, 40, 40), font=small_font)
        y += 62
    draw.line([285, 560, 615, 560], fill=(20, 20, 20), width=3)
    draw.text((290, 598), "TOTAL", fill=(20, 20, 20), font=text_font)
    draw.text((500, 598), "13.10", fill=(20, 20, 20), font=text_font)
    samples.append(
        ToySample(
            "receipt_total",
            img,
            "What are the TAX and TOTAL amounts printed on the receipt? Answer with both amounts.",
            "1.35 13.10",
            ["1.35", "13.10"],
        )
    )

    # 3) Count small objects in a grid. Aggressive compression tends to blur or
    # drop individual cells, causing count mistakes.
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((185, 75), "Count the blue dots", fill=(20, 20, 20), font=title_font)
    left, top, cell = 184, 170, 88
    blue_cells = {(0, 1), (1, 4), (2, 2), (3, 0), (3, 5), (4, 3), (5, 1)}
    red_cells = {(0, 5), (1, 1), (2, 4), (4, 0), (5, 5)}
    for r in range(6):
        for c in range(6):
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle([x0, y0, x0 + cell - 8, y0 + cell - 8], outline=(185, 185, 185), width=2)
            is_blue = (r, c) in blue_cells
            is_red = (r, c) in red_cells
            color = (35, 95, 220) if is_blue else (220, 70, 65) if is_red else (210, 210, 210)
            draw.ellipse([x0 + 25, y0 + 25, x0 + 54, y0 + 54], fill=color, outline=(30, 30, 30), width=1)
    samples.append(
        ToySample(
            "blue_dot_count",
            img,
            "How many blue dots and red dots are in the grid? Answer as blue count then red count.",
            "7 5",
            ["7", "5"],
        )
    )

    # 4) Read a tiny chart annotation rather than the obvious chart trend.
    img = Image.new("RGB", (image_size, image_size), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((115, 70), "Monthly micro chart", fill=(20, 20, 20), font=title_font)
    axis_x, axis_y = 150, 760
    draw.line([axis_x, 170, axis_x, axis_y], fill=(20, 20, 20), width=4)
    draw.line([axis_x, axis_y, 790, axis_y], fill=(20, 20, 20), width=4)
    bars = [
        ("red", (210, 65, 65), 330, "11"),
        ("green", (40, 150, 95), 455, "14"),
        ("purple", (126, 78, 190), 585, "17"),
        ("orange", (230, 145, 45), 410, "13"),
    ]
    x = 225
    for name, color, height, value in bars:
        draw.rectangle([x, axis_y - height, x + 85, axis_y], fill=color)
        draw.text((x + 20, axis_y - height - 43), value, fill=(25, 25, 25), font=small_font)
        draw.text((x - 5, axis_y + 18), name, fill=(25, 25, 25), font=small_font)
        x += 135
    samples.append(
        ToySample(
            "chart_value_lookup",
            img,
            "Which colored bar has the largest printed value, and what is that value? Answer color and value.",
            "purple 17",
            ["purple", "17"],
        )
    )

    # 5) Spatial relation among labeled objects. The answer is a small label, not
    # simply a visible object category.
    img = Image.new("RGB", (image_size, image_size), (238, 245, 255))
    draw = ImageDraw.Draw(img)
    draw.text((120, 80), "Find the label just left of the star", fill=(20, 20, 20), font=title_font)
    objects = [
        ("A1", (235, 360), "circle", (60, 150, 230)),
        ("B2", (405, 360), "square", (80, 180, 90)),
        ("C3", (575, 360), "star", (245, 195, 35)),
        ("D4", (745, 360), "triangle", (225, 85, 80)),
    ]
    for label, center, shape, color in objects:
        cx, cy = center
        if shape == "circle":
            draw.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=color, outline=(35, 35, 35), width=3)
        elif shape == "square":
            draw.rectangle([cx - 55, cy - 55, cx + 55, cy + 55], fill=color, outline=(35, 35, 35), width=3)
        elif shape == "triangle":
            draw.polygon([(cx, cy - 65), (cx - 65, cy + 55), (cx + 65, cy + 55)], fill=color, outline=(35, 35, 35))
        else:
            points = [
                (cx, cy - 68),
                (cx + 20, cy - 22),
                (cx + 70, cy - 18),
                (cx + 32, cy + 12),
                (cx + 44, cy + 62),
                (cx, cy + 36),
                (cx - 44, cy + 62),
                (cx - 32, cy + 12),
                (cx - 70, cy - 18),
                (cx - 20, cy - 22),
            ]
            draw.polygon(points, fill=color, outline=(35, 35, 35))
        draw.rounded_rectangle([cx - 43, cy + 85, cx + 43, cy + 134], radius=8, fill=(255, 255, 255), outline=(50, 50, 50), width=2)
        _draw_centered_text(draw, (cx - 43, cy + 85, cx + 43, cy + 134), label, tiny_code_font)
    samples.append(
        ToySample(
            "left_of_star_label",
            img,
            "What labels are under the objects immediately left and right of the yellow star? Answer both labels.",
            "B2 D4",
            ["b2", "d4"],
        )
    )

    # 6) Tiny map sign. This gives extra samples for non-quick benchmark runs.
    img = Image.new("RGB", (image_size, image_size), (245, 248, 242))
    draw = ImageDraw.Draw(img)
    draw.text((105, 80), "Terminal map", fill=(20, 20, 20), font=title_font)
    draw.line([120, 730, 790, 210], fill=(80, 80, 80), width=22)
    draw.line([150, 260, 760, 700], fill=(140, 140, 140), width=16)
    draw.rectangle([555, 165, 790, 265], fill=(250, 250, 210), outline=(30, 30, 30), width=4)
    _draw_centered_text(draw, (555, 165, 790, 265), "GATE C4", code_font)
    draw.ellipse([705, 270, 750, 315], fill=(220, 40, 40), outline=(35, 35, 35), width=3)
    _draw_centered_text(draw, (690, 318, 768, 370), "M1", tiny_code_font)
    samples.append(
        ToySample(
            "terminal_gate",
            img,
            "What gate code is on the top-right sign, and what label is next to the red marker? Answer both.",
            "C4 M1",
            ["c4", "m1"],
        )
    )

    # 7) Low-contrast serial number for full sweeps. It is deliberately hard and
    # should expose quality loss at very low retention ratios.
    img = Image.new("RGB", (image_size, image_size), (246, 246, 246))
    draw = ImageDraw.Draw(img)
    draw.text((120, 100), "Device back panel", fill=(45, 45, 45), font=title_font)
    draw.rectangle([170, 210, 725, 690], fill=(230, 233, 235), outline=(80, 80, 80), width=5)
    draw.rectangle([220, 325, 420, 395], fill=(238, 238, 238), outline=(110, 110, 110), width=3)
    draw.text((245, 342), "ZX-4", fill=(95, 95, 95), font=tiny_code_font)
    draw.rectangle([405, 505, 675, 585], fill=(238, 238, 238), outline=(110, 110, 110), width=3)
    draw.text((430, 522), "SN 8M2X", fill=(95, 95, 95), font=tiny_code_font)
    samples.append(
        ToySample(
            "low_contrast_serial",
            img,
            "What model code and serial number are on the gray labels? Answer both codes.",
            "ZX-4 8M2X",
            ["zx4", "8m2x"],
        )
    )

    # 8) Dense coordinate table with tiny alphanumeric values.
    img = Image.new("RGB", (image_size, image_size), (252, 252, 250))
    draw = ImageDraw.Draw(img)
    draw.text((120, 70), "Inventory lookup grid", fill=(20, 20, 20), font=title_font)
    rows = ["A", "B", "C", "D"]
    cols = ["1", "2", "3", "4"]
    values = [
        ["P3", "R6", "L8", "C2"],
        ["H5", "T1", "B7", "M4"],
        ["V2", "Q9", "Z9", "N6"],
        ["J8", "K3", "W5", "S1"],
    ]
    left, top, cell_w, cell_h = 215, 180, 135, 115
    for c, name in enumerate(cols):
        _draw_centered_text(draw, (left + c * cell_w, top - 65, left + (c + 1) * cell_w, top - 12), name, text_font)
    for r, name in enumerate(rows):
        _draw_centered_text(draw, (left - 70, top + r * cell_h, left - 12, top + (r + 1) * cell_h), name, text_font)
        for c in range(4):
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            fill = (245, 248, 255) if (r + c) % 2 == 0 else (255, 248, 235)
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=fill, outline=(85, 85, 85), width=2)
            _draw_centered_text(draw, (x0, y0, x0 + cell_w, y0 + cell_h), values[r][c], tiny_code_font)
    samples.append(
        ToySample(
            "inventory_grid_lookup",
            img,
            "Read the values in cells A2, B2, and C3 of the inventory grid. Answer the three values.",
            "R6 T1 Z9",
            ["r6", "t1", "z9"],
        )
    )

    # 9) Mini timetable with several plausible distractors.
    img = Image.new("RGB", (image_size, image_size), (245, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.text((135, 80), "Morning departures", fill=(20, 20, 20), font=title_font)
    headers = ["Route", "Time", "Platform"]
    table_left, table_top = 130, 180
    widths = [210, 240, 240]
    row_h = 82
    for c, header in enumerate(headers):
        x0 = table_left + sum(widths[:c])
        draw.rectangle([x0, table_top, x0 + widths[c], table_top + row_h], fill=(35, 75, 120), outline=(40, 40, 40), width=2)
        _draw_centered_text(draw, (x0, table_top, x0 + widths[c], table_top + row_h), header, text_font, fill=(255, 255, 255))
    routes = [("11", "08:20", "P1"), ("22", "09:45", "P3"), ("31", "10:05", "P2"), ("44", "11:30", "P5")]
    for r, row in enumerate(routes):
        y0 = table_top + (r + 1) * row_h
        for c, value in enumerate(row):
            x0 = table_left + sum(widths[:c])
            draw.rectangle([x0, y0, x0 + widths[c], y0 + row_h], fill=(255, 255, 255), outline=(90, 90, 90), width=2)
            _draw_centered_text(draw, (x0, y0, x0 + widths[c], y0 + row_h), value, text_font)
    samples.append(
        ToySample(
            "timetable_route_22",
            img,
            "For route 22, what time and platform are listed? Answer time and platform.",
            "09:45 P3",
            ["09:45", "p3"],
        )
    )

    # 10) Scatter plot: tiny point labels, not just color recognition.
    img = Image.new("RGB", (image_size, image_size), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((125, 75), "Labeled scatter points", fill=(20, 20, 20), font=title_font)
    draw.line([145, 790, 850, 790], fill=(25, 25, 25), width=4)
    draw.line([145, 790, 145, 165], fill=(25, 25, 25), width=4)
    points = [
        ("N4", (665, 255), (225, 65, 65)),
        ("R2", (420, 385), (225, 65, 65)),
        ("B8", (275, 690), (55, 105, 220)),
        ("D6", (720, 570), (55, 105, 220)),
        ("G1", (515, 500), (45, 150, 85)),
    ]
    for label, (x, y), color in points:
        draw.ellipse([x - 20, y - 20, x + 20, y + 20], fill=color, outline=(35, 35, 35), width=2)
        draw.text((x + 24, y - 25), label, fill=(20, 20, 20), font=small_font)
    samples.append(
        ToySample(
            "scatter_extreme_labels",
            img,
            "What is the label of the highest red point and the lowest blue point? Answer both labels.",
            "N4 B8",
            ["n4", "b8"],
        )
    )

    # 11) Medication-style label with many distractor numbers.
    img = Image.new("RGB", (image_size, image_size), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([145, 125, 875, 835], radius=28, fill=(255, 255, 255), outline=(40, 40, 40), width=5)
    draw.text((210, 170), "CLINIC SAMPLE", fill=(20, 20, 20), font=title_font)
    draw.text((210, 250), "Dose: 2 tablets", fill=(55, 55, 55), font=text_font)
    draw.text((210, 315), "Lot: HZ-41", fill=(55, 55, 55), font=text_font)
    draw.text((210, 380), "Expires: 04/27", fill=(55, 55, 55), font=text_font)
    draw.text((210, 445), "Storage: Room B12", fill=(55, 55, 55), font=text_font)
    draw.text((210, 510), "Ignore sample ID 9088", fill=(120, 120, 120), font=small_font)
    samples.append(
        ToySample(
            "clinic_label_expiry_room",
            img,
            "What expiration date and storage room are printed on the clinic sample label? Answer both.",
            "04/27 B12",
            ["04/27", "b12"],
        )
    )

    # 12) Shelf of vials: small codes tied to cap colors.
    img = Image.new("RGB", (image_size, image_size), (242, 246, 250))
    draw = ImageDraw.Draw(img)
    draw.text((145, 75), "Lab vial shelf", fill=(20, 20, 20), font=title_font)
    vial_specs = [
        ("blue", (70, 125, 225), "B-04"),
        ("green", (50, 160, 90), "RX-19"),
        ("amber", (215, 145, 45), "A7"),
        ("purple", (130, 80, 185), "P-6"),
    ]
    for i, (_name, cap_color, code) in enumerate(vial_specs):
        x = 170 + i * 175
        draw.rectangle([x, 245, x + 105, 685], fill=(235, 250, 255), outline=(55, 55, 55), width=3)
        draw.rectangle([x - 5, 205, x + 110, 255], fill=cap_color, outline=(55, 55, 55), width=3)
        draw.rectangle([x + 10, 505, x + 95, 585], fill=(255, 255, 255), outline=(80, 80, 80), width=2)
        _draw_centered_text(draw, (x + 10, 505, x + 95, 585), code, small_font)
    samples.append(
        ToySample(
            "vial_cap_codes",
            img,
            "What codes are on the vials with the green cap and the amber cap? Answer both codes.",
            "RX-19 A7",
            ["rx19", "a7"],
        )
    )

    # 13) Dense invoice line-item lookup.
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((165, 80), "Parts invoice", fill=(20, 20, 20), font=title_font)
    headers = ["Item", "Qty", "Price"]
    rows = [("cable", "5", "4.20"), ("adapter", "3", "18.75"), ("case", "1", "9.50"), ("clip", "8", "1.25")]
    x0, y0 = 150, 185
    widths = [315, 180, 220]
    for c, header in enumerate(headers):
        xx = x0 + sum(widths[:c])
        draw.rectangle([xx, y0, xx + widths[c], y0 + 78], fill=(35, 35, 35), outline=(35, 35, 35), width=2)
        _draw_centered_text(draw, (xx, y0, xx + widths[c], y0 + 78), header, text_font, fill=(255, 255, 255))
    for r, row in enumerate(rows):
        yy = y0 + (r + 1) * 78
        for c, value in enumerate(row):
            xx = x0 + sum(widths[:c])
            draw.rectangle([xx, yy, xx + widths[c], yy + 78], fill=(248, 248, 248), outline=(90, 90, 90), width=2)
            _draw_centered_text(draw, (xx, yy, xx + widths[c], yy + 78), value, text_font)
    samples.append(
        ToySample(
            "invoice_adapter_lookup",
            img,
            "For the adapter row, what quantity and price are shown? Answer quantity and price.",
            "3 18.75",
            ["3", "18.75"],
        )
    )

    # 14) Small map sign with two target words/numbers.
    img = Image.new("RGB", (image_size, image_size), (244, 250, 242))
    draw = ImageDraw.Draw(img)
    draw.text((140, 80), "Trail junction signs", fill=(20, 20, 20), font=title_font)
    draw.line([185, 780, 790, 250], fill=(110, 110, 110), width=20)
    draw.line([225, 255, 810, 760], fill=(165, 165, 165), width=16)
    green_sign = (575, 240, 805, 330)
    yellow_sign = (175, 665, 405, 755)
    draw.rounded_rectangle(green_sign, radius=12, fill=(175, 235, 175), outline=(35, 35, 35), width=4)
    draw.rounded_rectangle(yellow_sign, radius=12, fill=(255, 238, 130), outline=(35, 35, 35), width=4)
    _draw_centered_text(draw, green_sign, "EAST", code_font)
    _draw_centered_text(draw, yellow_sign, "72", code_font)
    samples.append(
        ToySample(
            "trail_signs",
            img,
            "What word is on the green sign and what number is on the yellow sign? Answer both.",
            "EAST 72",
            ["east", "72"],
        )
    )

    # 15) Phone-like notification card, useful for OCR and layout stress.
    img = Image.new("RGB", (image_size, image_size), (234, 238, 245))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([270, 105, 750, 895], radius=45, fill=(25, 28, 35), outline=(60, 60, 65), width=5)
    draw.rounded_rectangle([315, 210, 705, 325], radius=25, fill=(245, 245, 245), outline=(180, 180, 180), width=2)
    draw.text((345, 230), "AL", fill=(25, 25, 25), font=title_font)
    draw.text((430, 238), "Code 5821", fill=(25, 25, 25), font=text_font)
    draw.rounded_rectangle([315, 370, 705, 485], radius=25, fill=(245, 245, 245), outline=(180, 180, 180), width=2)
    draw.text((345, 390), "BK", fill=(25, 25, 25), font=title_font)
    draw.text((430, 398), "Code 7440", fill=(95, 95, 95), font=text_font)
    samples.append(
        ToySample(
            "phone_notification_code",
            img,
            "In the top notification, what sender initials and code are shown? Answer initials and code.",
            "AL 5821",
            ["al", "5821"],
        )
    )

    return samples


def build_multi_image_case(
    dataset: Sequence[ToySample],
    start_index: int,
    num_images: int,
) -> Dict[str, Any]:
    selected = [dataset[(start_index + offset) % len(dataset)] for offset in range(num_images)]
    images = [sample.image for sample in selected]
    if num_images == 1:
        question = selected[0].question + " Keep the answer short."
    else:
        per_image_questions = [
            f"Image {idx + 1}: {sample.question}"
            for idx, sample in enumerate(selected)
        ]
        question = (
            f"You are given {num_images} images. Answer each image-specific question in order. "
            "Return only the requested values, separated by semicolons. "
            + " ".join(per_image_questions)
        )
    keywords: List[str] = []
    for sample in selected:
        keywords.extend(sample.keywords)
    reference_answer = " ".join(keywords)
    return {
        "sample_id": "+".join(sample.sample_id for sample in selected),
        "images": images,
        "question": question,
        "reference_answer": reference_answer,
        "keywords": keywords,
    }


class BenchmarkRunner:
    def __init__(self, engine: VLMEngine, config: Dict[str, Any]) -> None:
        self.engine = engine
        self.config = config
        self.benchmark_config = config.get("benchmark", {})
        self.compression_config = config.get("compression", {})
        self.quality_config = config.get("quality", {})

    def _total_runs(
        self,
        methods: Sequence[str],
        ratios: Sequence[float],
        resolutions: Sequence[str],
        num_images_values: Sequence[int],
        max_samples: int,
    ) -> int:
        method_ratio_count = 0
        for method in methods:
            method_ratio_count += 1 if method == "none" else len(ratios)
        return method_ratio_count * len(resolutions) * len(num_images_values) * max_samples

    def run(
        self,
        methods: Sequence[str] | None = None,
        ratios: Sequence[float] | None = None,
        resolutions: Sequence[str] | None = None,
        num_images_values: Sequence[int] | None = None,
        max_samples: int | None = None,
        output_csv: str | Path | None = None,
        save_every: int = 1,
    ) -> pd.DataFrame:
        methods = list(methods or self.benchmark_config.get("methods", ["none", "fixed", "importance", "merging"]))
        ratios = [float(x) for x in (ratios or self.benchmark_config.get("retention_ratios", [1.0, 0.5, 0.25]))]
        resolutions = list(resolutions or self.benchmark_config.get("image_resolutions", ["medium"]))
        num_images_values = [int(x) for x in (num_images_values or self.benchmark_config.get("num_images", [1]))]
        max_samples = int(max_samples or self.benchmark_config.get("max_samples", 3))
        output_csv = Path(output_csv or self.benchmark_config.get("output_csv", "results/benchmark_results.csv"))
        ensure_dir(output_csv.parent)

        dataset = create_toy_dataset()
        max_samples = min(max_samples, len(dataset))
        records: List[Dict[str, Any]] = []
        total = self._total_runs(methods, ratios, resolutions, num_images_values, max_samples)
        metric = self.quality_config.get("metric", "keyword_match")
        apply_proxy = bool(self.compression_config.get("apply_proxy_image_budget", True))
        max_new_tokens = int(self.config.get("generation", {}).get("max_new_tokens", 64))
        warmup_runs = int(self.benchmark_config.get("warmup_runs", 1))

        if warmup_runs > 0:
            warmup_case = build_multi_image_case(dataset, 0, 1)
            warmup_tokens = min(max_new_tokens, 16)
            print(f"Running {warmup_runs} warmup inference(s), not recorded...")
            for _ in range(warmup_runs):
                try:
                    self.engine.generate_answer(
                        image=warmup_case["images"],
                        question=warmup_case["question"],
                        compression_method="none",
                        retention_ratio=1.0,
                        image_resolution=resolutions[0],
                        max_new_tokens=warmup_tokens,
                    )
                except Exception as exc:
                    print(f"Warmup failed; continuing to recorded benchmark. Error: {exc}")
                    break

        progress = tqdm(total=total, desc="Benchmark")
        for method_name in methods:
            method_ratios = [1.0] if method_name == "none" else ratios
            for ratio in method_ratios:
                compression_method = create_compression_method(
                    method_name,
                    retention_ratio=ratio,
                    apply_proxy_image_budget=apply_proxy,
                )
                for image_resolution in resolutions:
                    for num_images in num_images_values:
                        for sample_idx in range(max_samples):
                            case = build_multi_image_case(dataset, sample_idx, num_images)
                            record: Dict[str, Any] = {
                                "model_id": self.engine.model_id,
                                "compression_method": compression_method.name,
                                "retention_ratio": compression_method.retention_ratio,
                                "input_resolution": image_resolution,
                                "num_images": num_images,
                                "max_new_tokens": max_new_tokens,
                                "sample_id": case["sample_id"],
                                "question": case["question"],
                                "reference_answer": case["reference_answer"],
                                "keywords": ",".join(case["keywords"]),
                                "quality_metric": metric,
                                "success": False,
                                "oom": False,
                                "error": "",
                                "error_traceback": "",
                                "proxy_image_budget": apply_proxy,
                                "generated_answer": "",
                                "latency_ms": None,
                                "peak_gpu_memory_mb": None,
                                "throughput_tokens_per_second": None,
                                "generated_tokens": None,
                                "quality_score": None,
                                "number_of_visual_tokens": None,
                                "original_visual_tokens": None,
                                "kept_visual_tokens": None,
                                "compression_applied_internal": False,
                                "original_seq_len": None,
                                "compressed_seq_len": None,
                            }
                            try:
                                reset_peak_gpu_memory()
                                result = self.engine.generate_answer(
                                    image=case["images"],
                                    question=case["question"],
                                    compression_method=compression_method,
                                    retention_ratio=ratio,
                                    image_resolution=image_resolution,
                                    max_new_tokens=max_new_tokens,
                                )
                                peak_memory = get_peak_gpu_memory_mb()
                                quality = compute_quality_score(
                                    result["generated_answer"],
                                    case["reference_answer"],
                                    keywords=case["keywords"],
                                    metric=metric,
                                )
                                record.update(result)
                                record.update(
                                    {
                                        "peak_gpu_memory_mb": peak_memory,
                                        "quality_score": quality,
                                        "success": True,
                                    }
                                )
                            except torch.cuda.OutOfMemoryError as exc:
                                record.update(
                                    {
                                        "oom": True,
                                        "error": str(exc),
                                        "error_traceback": traceback.format_exc(),
                                    }
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except RuntimeError as exc:
                                message = str(exc)
                                record.update(
                                    {
                                        "oom": "out of memory" in message.lower(),
                                        "error": message,
                                        "error_traceback": traceback.format_exc(),
                                    }
                                )
                                if record["oom"] and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception as exc:
                                record.update({"error": repr(exc), "error_traceback": traceback.format_exc()})

                            records.append(record)
                            if save_every > 0 and len(records) % save_every == 0:
                                pd.DataFrame(records).to_csv(output_csv, index=False)
                            progress.update(1)

        progress.close()
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)

        summary_csv = self.benchmark_config.get("summary_csv")
        if summary_csv:
            summarize_results(df, output_csv=summary_csv)
        return df


def run_benchmark(engine: VLMEngine, config: Dict[str, Any], **kwargs: Any) -> pd.DataFrame:
    return BenchmarkRunner(engine, config).run(**kwargs)
