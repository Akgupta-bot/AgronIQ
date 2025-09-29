# services/report_generator.py
from fpdf import FPDF
from datetime import datetime
import io
import os
import urllib.request
import pandas as pd
import numpy as np
import tempfile
import textwrap
from typing import Optional

# Font directory
FONT_DIR = os.path.join("services", "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

FONT_FILE = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"


def ensure_font():
    """Download DejaVuSans.ttf if missing."""
    if os.path.exists(FONT_FILE):
        return
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_FILE)
    except Exception:
        return


def safe_text(txt: str) -> str:
    """Clean text and replace line breaks."""
    if txt is None:
        return "N/A"
    if not isinstance(txt, str):
        txt = str(txt)
    return txt.replace("\n", " ").replace("\r", " ")


def sanitize_for_pdf(txt: str, font_ok: bool, width: int = 90) -> str:
    """Prepare text for PDF rendering."""
    s = safe_text(str(txt))
    if font_ok:
        s = "".join(ch if ord(ch) <= 0xFFFF else "?" for ch in s)
    else:
        s = s.encode("latin-1", "replace").decode("latin-1")
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s


def _safe_val(val, max_len: int = 40):
    """Convert values safely to string for PDF."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if isinstance(val, (list, tuple, np.ndarray)):
            val = ", ".join(map(str, val))
        s = str(val).strip()
        if len(s) > max_len:
            s = s[:max_len] + "..."
        return s
    except Exception:
        return "N/A"


def _pdf_write_heading(pdf: FPDF, text: str, font_ok: bool):
    """Write a bold heading line."""
    if font_ok:
        pdf.set_font("DejaVu", size=14)
    else:
        pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, sanitize_for_pdf(text, font_ok), ln=True, align="L")


def _pdf_write_line(pdf: FPDF, text: str, font_ok: bool):
    """Safe line writer: avoids FPDF width errors."""
    try:
        clean_text = sanitize_for_pdf(text, font_ok)
        if not clean_text.strip():
            clean_text = "N/A"

        if len(clean_text) < 40 and "\n" not in clean_text:
            pdf.cell(0, 6, clean_text, ln=True)
        else:
            pdf.multi_cell(0, 6, clean_text)
    except Exception:
        pdf.cell(0, 6, "N/A", ln=True)


def _pdf_write_table(pdf: FPDF, df: pd.DataFrame, font_ok: bool, max_rows: int = 5):
    """Render dataframe preview as table."""
    if df.empty:
        _pdf_write_line(pdf, "No data available.", font_ok)
        return

    df = df.head(max_rows).copy().astype(str)

    col_count = len(df.columns)
    col_width = max(20, (pdf.w - 20) / col_count - 2)  # ensure min width
    row_height = pdf.font_size * 1.5

    # Header
    pdf.set_fill_color(200, 200, 200)
    for col in df.columns:
        pdf.cell(col_width, row_height, sanitize_for_pdf(col, font_ok, width=20), border=1, align="C", fill=True)
    pdf.ln(row_height)

    # Rows
    for _, row in df.iterrows():
        for col in df.columns:
            val = sanitize_for_pdf(str(row[col]), font_ok, width=20)
            pdf.cell(col_width, row_height, val, border=1, align="C")
        pdf.ln(row_height)


def create_pdf_report(
    crop_img_bytes: Optional[bytes],
    soil_df: Optional[pd.DataFrame],
    weather_df: Optional[pd.DataFrame],
    pest_df: Optional[pd.DataFrame],
    recommendations: Optional[list],
) -> io.BytesIO:
    """Generate a PDF report with crop, soil, weather, pest, and AI recommendations."""
    soil_df = soil_df if isinstance(soil_df, pd.DataFrame) else pd.DataFrame()
    weather_df = weather_df if isinstance(weather_df, pd.DataFrame) else pd.DataFrame()
    pest_df = pest_df if isinstance(pest_df, pd.DataFrame) else pd.DataFrame()
    recommendations = recommendations or []

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()

    # Font setup
    font_ok = False
    try:
        ensure_font()
        if os.path.exists(FONT_FILE):
            pdf.add_font("DejaVu", "", FONT_FILE, uni=True)
            font_ok = True
    except Exception:
        font_ok = False

    # Title
    if font_ok:
        pdf.set_font("DejaVu", size=16)
    else:
        pdf.set_font("Arial", "B", 16)
    _pdf_write_line(pdf, "AgronIQ - Crop & Soil Analysis Report", font_ok)
    pdf.ln(6)

    # Summary
    if font_ok:
        pdf.set_font("DejaVu", size=11)
    else:
        pdf.set_font("Arial", size=11)
    _pdf_write_line(pdf, f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", font_ok)
    pdf.ln(4)

    # Crop Image
    if crop_img_bytes:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(crop_img_bytes)
                tmp_path = tmp.name
            pdf.image(tmp_path, w=90)
            pdf.ln(4)
        except Exception:
            pass
        finally:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    # Soil
    _pdf_write_heading(pdf, "Soil KPIs (Latest)", font_ok)
    try:
        latest = soil_df.iloc[-1]
        _pdf_write_line(pdf, f"Soil Moisture: {_safe_val(latest.get('SoilMoisture(%)'))}%", font_ok)
        _pdf_write_line(pdf, f"Soil Temp: {_safe_val(latest.get('SoilTemp(C)'))} °C", font_ok)
        _pdf_write_line(pdf, f"pH: {_safe_val(latest.get('pH'))}", font_ok)
        _pdf_write_line(pdf,
            f"N: {_safe_val(latest.get('Nitrogen(mg/kg)'))}, "
            f"P: {_safe_val(latest.get('Phosphorus(mg/kg)'))}, "
            f"K: {_safe_val(latest.get('Potassium(mg/kg)'))}", font_ok
        )
    except Exception:
        _pdf_write_line(pdf, "Soil data unavailable.", font_ok)
    pdf.ln(4)

    # Weather
    _pdf_write_heading(pdf, "Weather (Latest)", font_ok)
    try:
        w = weather_df.iloc[-1]
        _pdf_write_line(pdf, f"Rainfall: {_safe_val(w.get('Rainfall(mm)', w.get('Rain(mm)', 'N/A')))} mm", font_ok)
        _pdf_write_line(pdf, f"Temperature: {_safe_val(w.get('Temp(C)', w.get('Temperature', 'N/A')))} °C", font_ok)
        _pdf_write_line(pdf, f"Humidity: {_safe_val(w.get('Humidity(%)', w.get('Humidity', 'N/A')))}%", font_ok)
    except Exception:
        _pdf_write_line(pdf, "Weather data unavailable.", font_ok)
    pdf.ln(4)

    # Pest Risk
    _pdf_write_heading(pdf, "Pest Risk Summary", font_ok)
    try:
        p = pest_df.iloc[-1]
        _pdf_write_line(pdf, f"Risk Level: {_safe_val(p.get('RiskLevel'))} {_safe_val(p.get('Symbol'))}", font_ok)
        _pdf_write_line(pdf, f"Reasons: {_safe_val(p.get('Reasons'))}", font_ok)
    except Exception:
        _pdf_write_line(pdf, "Pest data unavailable.", font_ok)
    pdf.ln(6)

    # Recommendations
    _pdf_write_heading(pdf, "AI Recommendations", font_ok)
    if recommendations:
        for rec in recommendations:
            _pdf_write_line(pdf, f"- {_safe_val(rec)}", font_ok)
    else:
        _pdf_write_line(pdf, "No recommendations generated.", font_ok)

    # Data Previews
    if not soil_df.empty:
        pdf.ln(4)
        _pdf_write_heading(pdf, "Soil Data (Preview)", font_ok)
        _pdf_write_table(pdf, soil_df, font_ok)
    if not weather_df.empty:
        pdf.ln(4)
        _pdf_write_heading(pdf, "Weather Data (Preview)", font_ok)
        _pdf_write_table(pdf, weather_df, font_ok)

    # Export
    buf = io.BytesIO()
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, (bytes, bytearray)):
        buf.write(pdf_bytes)
    elif isinstance(pdf_bytes, str):
        buf.write(pdf_bytes.encode("latin-1", "replace"))
    else:
        try:
            buf.write(bytes(pdf_bytes))
        except Exception:
            buf.write(b"")
    buf.seek(0)
    return buf
