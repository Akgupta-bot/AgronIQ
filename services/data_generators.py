# services/data_generators.py
import numpy as np
import pandas as pd


def generate_crop_image(H=128, W=128, seed=42):
    """
    Generate a synthetic multispectral crop image with:
    - RGB + NIR + RedEdge + SWIR bands
    - NDVI, NDWI, SAVI, GCI indices
    - Stress labels (0=healthy, 1=stressed)
    Returns dict with arrays and flattened dataset (X, y).
    """
    np.random.seed(int(seed))
    yy, xx = np.indices((H, W))
    center = (H // 2, W // 2)
    r = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)

    # Base reflectance
    red = 0.20 + 0.05 * np.random.randn(H, W)
    green = 0.30 + 0.05 * np.random.randn(H, W)
    blue = 0.25 + 0.05 * np.random.randn(H, W)
    nir = 0.60 + 0.07 * np.random.randn(H, W)
    red_edge = 0.55 + 0.06 * np.random.randn(H, W)
    swir = 0.40 + 0.08 * np.random.randn(H, W)

    # Healthy patch (center)
    mask_healthy = r < 36
    nir[mask_healthy] = 0.78
    green[mask_healthy] = 0.42
    red_edge[mask_healthy] = 0.65

    # Stressed patch (ring)
    mask_stressed = (r > 60) & (r < 84)
    red[mask_stressed] = 0.40
    nir[mask_stressed] = 0.28
    red_edge[mask_stressed] = 0.35

    # Random small stress spots
    for _ in range(12):
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        rad = np.random.randint(3, 14)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < rad
        red[rr] = 0.4
        nir[rr] = 0.28
        red_edge[rr] = 0.33

    # Clip values
    for a in (red, green, blue, nir, red_edge, swir):
        np.clip(a, 0.0, 1.0, out=a)

    # Vegetation indices (safe denominator)
    ndvi = (nir - red) / np.clip((nir + red), 1e-6, None)
    ndwi = (green - nir) / np.clip((green + nir), 1e-6, None)
    savi = ((nir - red) / np.clip((nir + red + 0.5), 1e-6, None)) * 1.5
    gci = (nir / np.clip(green, 1e-6, None)) - 1

    # Labels (0 healthy, 1 stressed)
    labels = np.zeros((H, W), dtype=int)
    labels[mask_stressed] = 1

    # Flattened dataset
    X = np.stack([
        red.ravel(), green.ravel(), blue.ravel(),
        nir.ravel(), red_edge.ravel(), swir.ravel(),
        ndvi.ravel(), ndwi.ravel(), savi.ravel(), gci.ravel()
    ], axis=1).astype(np.float32)
    y = labels.ravel().astype(int)

    return {
        "red": red, "green": green, "blue": blue,
        "nir": nir, "red_edge": red_edge, "swir": swir,
        "ndvi": ndvi, "ndwi": ndwi, "savi": savi, "gci": gci,
        "labels": labels, "X": X, "y": y
    }


def generate_soil_data(days=30, seed=42, inject_anomalies=True):
    """
    Generate synthetic soil dataset (moisture, temp, pH, nutrients).
    Optionally inject anomalies for testing.
    """
    np.random.seed(int(seed))
    df = pd.DataFrame({
        "Day": np.arange(1, days + 1),
        "SoilMoisture(%)": np.random.randint(20, 90, size=days).astype(float),
        "SoilTemp(C)": np.random.randint(15, 35, size=days).astype(float),
        "pH": np.round(np.random.uniform(5.5, 8.0, size=days), 2),
        "Nitrogen(mg/kg)": np.random.randint(100, 300, size=days).astype(float),
        "Phosphorus(mg/kg)": np.random.randint(20, 100, size=days).astype(float),
        "Potassium(mg/kg)": np.random.randint(50, 200, size=days).astype(float),
        "Anomaly": np.zeros(days, dtype=int),
    })

    if inject_anomalies and days >= 2:
        idx = np.random.choice(days, 2, replace=False)
        df.loc[idx[0], ["SoilMoisture(%)", "pH", "Anomaly"]] = [0.0, 3.5, 1]
        df.loc[idx[1], ["SoilMoisture(%)", "pH", "Anomaly"]] = [120.0, 9.5, 1]

    return df


def generate_weather_data(days=30, seed=99, inject_anomalies=True):
    """
    Generate synthetic weather dataset (rainfall, temp, humidity).
    Optionally inject anomalies for testing.
    """
    np.random.seed(int(seed))
    df = pd.DataFrame({
        "Day": np.arange(1, days + 1),
        "Rainfall(mm)": np.random.randint(0, 50, size=days).astype(float),
        "Temp(C)": np.random.randint(18, 38, size=days).astype(float),
        "Humidity(%)": np.random.randint(40, 95, size=days).astype(float),
        "Anomaly": np.zeros(days, dtype=int),
    })

    if inject_anomalies and days >= 2:
        idx = np.random.choice(days, 2, replace=False)
        df.loc[idx[0], ["Temp(C)", "Humidity(%)", "Anomaly"]] = [5.0, 5.0, 1]
        df.loc[idx[1], ["Temp(C)", "Humidity(%)", "Anomaly"]] = [50.0, 120.0, 1]

    return df
