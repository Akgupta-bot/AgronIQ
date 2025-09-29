# pages/technician.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime


def app():
    st.header(f"🔧 Technician Dashboard — Welcome, {st.session_state['username']}")
    st.caption("Keep IoT devices healthy, validate field data, and log offline entries ⚡")

    # -------------------------------
    # Session state init
    # -------------------------------
    if "offline_entries" not in st.session_state:
        st.session_state["offline_entries"] = []

    # -------------------------------
    # Tabs
    # -------------------------------
    tabs = st.tabs(
        [
            "📡 Device Fleet",
            "🗺️ Map View",
            "📸 Validation Upload",
            "📝 Offline Entry",
        ]
    )

    # --------------------------------
    # Tab 1: Device Fleet
    # --------------------------------
    with tabs[0]:
        st.subheader("📡 Device Fleet Status")

        ns = st.slider("Number of devices to simulate", 1, 12, 5)
        sensors, alerts = [], []

        for i in range(ns):
            battery = np.random.randint(10, 100)
            lastsync = np.random.randint(1, 180)
            dev = {
                "DeviceID": f"S-{100+i}",
                "Latitude": 20 + np.random.rand() * 0.5,
                "Longitude": 75 + np.random.rand() * 0.5,
                "Battery(%)": battery,
                "LastSync(min)": lastsync,
            }
            sensors.append(dev)

            # --- alert conditions ---
            if battery < 20:
                alerts.append(f"⚠️ {dev['DeviceID']} battery low ({battery}%)")
            if lastsync > 120:
                alerts.append(f"📡 {dev['DeviceID']} not synced in {lastsync} min")

        sdf = pd.DataFrame(sensors)
        st.dataframe(sdf, use_container_width=True)

        # Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Devices", len(sdf))
        with c2:
            st.metric("⚡ Healthy", f"{100 * (1 - len(alerts)/len(sdf)):.1f}%")
        with c3:
            st.metric("⚠️ Alerts", len(alerts))

        if alerts:
            st.error("⚠️ Real-Time Alerts Detected")
            for a in alerts:
                st.markdown(f"- {a}")
        else:
            st.success("✅ All devices healthy")

    # --------------------------------
    # Tab 2: Map
    # --------------------------------
    with tabs[1]:
        st.subheader("🗺️ Sensor Map")
        try:
            st.map(sdf.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}))
        except Exception:
            st.info("ℹ️ No device data available to plot. Run fleet simulation first.")

    # --------------------------------
    # Tab 3: Validation Upload
    # --------------------------------
    with tabs[2]:
        st.subheader("📸 Field Image Validation")

        vimg = st.file_uploader("Upload Validation Photo", type=["jpg", "png"])
        notes = st.text_area("📝 Technician Notes", placeholder="Enter observations about the field...")

        if vimg and st.button("💾 Save Validation Entry", use_container_width=True):
            imgv = Image.open(vimg)
            st.image(imgv, caption="Uploaded Field Image", use_container_width=True)
            st.success("✅ Validation entry saved")
            if notes.strip():
                st.info(f"📌 Notes: *{notes}*")

    # --------------------------------
    # Tab 4: Offline Data Entry
    # --------------------------------
    with tabs[3]:
        st.subheader("📝 Offline Sensor Data Entry")

        with st.form("offline_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                d_day = st.number_input("Day", min_value=1, max_value=365, value=1)
                d_moist = st.number_input("SoilMoisture(%)", min_value=0, max_value=100, value=40)
            with col2:
                d_temp = st.number_input("SoilTemp(C)", min_value=-10, max_value=60, value=25)
                d_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
            with col3:
                d_n = st.number_input("Nitrogen(mg/kg)", min_value=0, max_value=1000, value=150)
                d_p = st.number_input("Phosphorus(mg/kg)", min_value=0, max_value=500, value=40)
                d_k = st.number_input("Potassium(mg/kg)", min_value=0, max_value=1000, value=100)

            submit = st.form_submit_button("💾 Save Offline Reading")

            if submit:
                entry = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Day": d_day,
                    "SoilMoisture(%)": d_moist,
                    "SoilTemp(C)": d_temp,
                    "pH": d_ph,
                    "Nitrogen(mg/kg)": d_n,
                    "Phosphorus(mg/kg)": d_p,
                    "Potassium(mg/kg)": d_k,
                }
                st.session_state["offline_entries"].append(entry)

                st.success(f"✅ Reading saved for Day {d_day}")

                # --- anomaly detection ---
                anomalies = []
                if d_moist < 15 or d_moist > 85:
                    anomalies.append("💧 Soil moisture unusually high/low")
                if d_temp < 5 or d_temp > 45:
                    anomalies.append("🌡️ Soil temperature outside expected range")
                if d_ph < 5.5 or d_ph > 8.5:
                    anomalies.append("⚗️ Soil pH outside optimal range")

                if anomalies:
                    st.error("⚠️ Anomalies detected:")
                    for a in anomalies:
                        st.markdown(f"- {a}")

        # --- show history ---
        if st.session_state["offline_entries"]:
            st.markdown("### 📑 Saved Offline Entries")
            with st.expander("Show History", expanded=False):
                st.dataframe(pd.DataFrame(st.session_state["offline_entries"]), use_container_width=True)
