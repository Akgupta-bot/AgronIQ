# AgronIQ — AI Crop & Soil Monitoring Prototype

## Overview
AgronIQ is a Streamlit demo prototype for crop/soil monitoring with:
- Farmer, Agronomist, Researcher, Technician dashboards
- Multispectral image simulation & index visualization (NDVI, NDWI, SAVI)
- AI models: RandomForest / MLP and optional CNN (TensorFlow)
- Anomaly detection on time-series (IsolationForest)
- Report generation (PDF)
- Simple chat/advisory snapshot

## Quick start
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
# optionally: pip install tensorflow
