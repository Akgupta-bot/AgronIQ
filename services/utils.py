import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import datetime

# ---------------------------
# Soil Data Helpers
# ---------------------------

def standardize_soil_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize soil dataframe to required schema.
    Ensures consistent columns and fills missing with NaN.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    colmap = {
        "day": "Day",
        "moisture": "SoilMoisture(%)",
        "soilmoisture": "SoilMoisture(%)",
        "soilmoisture(%)": "SoilMoisture(%)",
        "sm": "SoilMoisture(%)",
        "temp": "SoilTemp(C)",
        "soiltemp": "SoilTemp(C)",
        "soiltemp(c)": "SoilTemp(C)",
        "temperature": "SoilTemp(C)",
        "ph": "pH",
        "n": "Nitrogen(mg/kg)",
        "nitrogen": "Nitrogen(mg/kg)",
        "p": "Phosphorus(mg/kg)",
        "phosphorus": "Phosphorus(mg/kg)",
        "k": "Potassium(mg/kg)",
        "potassium": "Potassium(mg/kg)",
    }

    df = df.rename(columns=lambda c: colmap.get(c, c))

    required_cols = [
        "Day",
        "SoilMoisture(%)",
        "SoilTemp(C)",
        "pH",
        "Nitrogen(mg/kg)",
        "Phosphorus(mg/kg)",
        "Potassium(mg/kg)",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[required_cols]


def plot_soil_trends(soil_df: pd.DataFrame):
    """Plot soil moisture and temperature trends (robust to missing columns)."""
    soil_df = standardize_soil_df(soil_df)
    available = [c for c in ["SoilMoisture(%)", "SoilTemp(C)"] if c in soil_df.columns]

    if not available:
        return px.line(title="No valid soil moisture/temperature columns found")

    df_long = soil_df.melt(id_vars="Day", value_vars=available,
                           var_name="Parameter", value_name="Value")
    return px.line(df_long, x="Day", y="Value", color="Parameter",
                   title="Soil Moisture & Temperature Trends")


def plot_soil_nutrients(soil_df: pd.DataFrame):
    """Plot soil nutrient levels (N, P, K) with robust column checks."""
    soil_df = standardize_soil_df(soil_df)
    available = [c for c in ["Nitrogen(mg/kg)", "Phosphorus(mg/kg)", "Potassium(mg/kg)"] if c in soil_df.columns]

    if not available:
        return px.line(title="No valid NPK columns found")

    df_long = soil_df.melt(id_vars="Day", value_vars=available,
                           var_name="Nutrient", value_name="Value")
    return px.line(df_long, x="Day", y="Value", color="Nutrient",
                   title="Soil Nutrient Trends (NPK)")


# ---------------------------
# Crop Image Analysis Helpers
# ---------------------------

def compute_indices(arr_rgb):
    """
    Compute vegetation indices from RGB (and pseudo NIR).
    Returns dict of indices.
    """
    red = arr_rgb[:, :, 0]
    green = arr_rgb[:, :, 1]
    blue = arr_rgb[:, :, 2]

    # Approximate NIR using red+green (if no NIR provided)
    nir = red * 0.6 + green * 0.4

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)  # water index
    savi = ((nir - red) / (nir + red + 0.5)) * 1.5
    gci = (nir / (green + 1e-6)) - 1  # chlorophyll index

    return {"NDVI": ndvi, "NDWI": ndwi, "SAVI": savi, "GCI": gci}


def plot_crop_analysis(arr_rgb, ndvi, preds):
    """Show RGB, NDVI, and prediction maps side-by-side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr_rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    im1 = axes[1].imshow(ndvi, cmap="RdYlGn")
    axes[1].set_title("NDVI")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(preds, cmap="coolwarm")
    axes[2].set_title("Stress Prediction")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    st.pyplot(fig)


# ---------------------------
# Anomaly Detection
# ---------------------------

def detect_anomalies(df: pd.DataFrame, column: str, z_thresh: float = 2.5):
    """
    Detect anomalies in a numeric column using z-score.
    Returns DataFrame with 'Anomaly' flag.
    """
    if column not in df.columns:
        return df

    series = df[column].astype(float)
    z_scores = (series - series.mean()) / (series.std() + 1e-6)
    df["Anomaly"] = (np.abs(z_scores) > z_thresh)
    return df


# ---------------------------
# Pest Risk Helpers
# ---------------------------

def pest_risk_score(weather_row, soil_moisture):
    """Compute pest risk score from weather + soil conditions."""
    score, reasons = 0, []

    rain = weather_row.get("Rainfall(mm)", weather_row.get("Rain(mm)", 0))
    temp = weather_row.get("Temp(C)", weather_row.get("Temperature", 0))
    humidity = weather_row.get("Humidity(%)", weather_row.get("Humidity", 0))

    if humidity > 80:
        score += 20; reasons.append("High humidity")
    if temp > 30:
        score += 20; reasons.append("High temperature")
    if soil_moisture > 60:
        score += 15; reasons.append("High soil moisture")
    if rain > 5:
        score += 15; reasons.append("Recent rain")

    return min(score, 100), reasons


def pest_risk_level(score):
    """Convert risk score to qualitative level + emoji."""
    if score < 30:
        return "Low", "🟢"
    elif score < 60:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


# ---------------------------
# Chat Helpers
# ---------------------------

def add_chat_entry(sender, message="", file_name=None, file_bytes=None):
    """Add a message (or file) to session_state chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "sender": sender,
        "message": message,
        "file_name": file_name,
        "file_bytes": file_bytes,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
    })


# ---------------------------
# Session Init
# ---------------------------

def init_session_state(st_ref):
    """Initialize all common session_state variables."""
    if "chat_history" not in st_ref.session_state:
        st_ref.session_state.chat_history = []
    if "linked_devices" not in st_ref.session_state:
        st_ref.session_state["linked_devices"] = []
    if "device_data" not in st_ref.session_state:
        st_ref.session_state["device_data"] = {}
