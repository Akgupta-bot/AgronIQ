# pages/agronomist.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io
import base64
import os
from PIL import Image
from services import data_generators, utils
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ------------------------
# Utility Functions
# ------------------------
def compute_spectral_indices(crop):
    """Compute NDVI, NDWI, SAVI, GCI from synthetic crop bands (uppercase keys)."""
    red, green, blue, nir = crop["RED"], crop["GREEN"], crop["BLUE"], crop["NIR"]

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    savi = (1.5 * (nir - red)) / (nir + red + 0.5 + 1e-6)
    gci = (nir / (green + 1e-6)) - 1.0

    return {
        "NDVI": float(ndvi.mean()),
        "NDWI": float(ndwi.mean()),
        "SAVI": float(savi.mean()),
        "GCI": float(gci.mean()),
    }, {"NDVI": ndvi, "NDWI": ndwi, "SAVI": savi, "GCI": gci}


def detect_soil_anomalies(soil_df: pd.DataFrame):
    """Detect anomalies in soil data (very high/low values)."""
    anomalies = {}
    if soil_df.empty:
        return anomalies

    anomalies["Moisture"] = soil_df[
        soil_df["SoilMoisture(%)"].between(0, 15) |
        soil_df["SoilMoisture(%)"].between(85, 100)
    ]
    anomalies["pH"] = soil_df[(soil_df["pH"] < 5.5) | (soil_df["pH"] > 8.0)]
    anomalies["Nutrients"] = soil_df[
        (soil_df["Nitrogen(mg/kg)"] < 80) |
        (soil_df["Phosphorus(mg/kg)"] < 15) |
        (soil_df["Potassium(mg/kg)"] < 40)
    ]
    return anomalies


# ------------------------
# Main App
# ------------------------
def app():
    st.title(f"🌾 Agronomist Dashboard — Welcome, {st.session_state['username']}")
    st.caption("Turning Field Data into Smart Decisions 💡")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tabs = st.tabs(["🌾 Plot Comparison", "🛰️ Device Data", "📝 Advisory", "💬 Chat"])

    # -------------------------
    # Tab 1: Plot Comparison
    # -------------------------
    with tabs[0]:
        st.markdown("### 🔎 Compare Farmer Plots")

        num_plots = st.slider("Select number of plots", 2, 6, 3)
        seed = st.number_input("Base seed", min_value=1, max_value=9999, value=101)

        plots = []
        cols = st.columns(num_plots)

        for i in range(num_plots):
            crop_i = data_generators.generate_crop_image(seed=seed + i * 7)
            soil_i = data_generators.generate_soil_data(seed=seed + i * 7)
            plots.append((crop_i, soil_i))

            with cols[i]:
                st.image(
                    (np.dstack([crop_i["RED"], crop_i["GREEN"], crop_i["BLUE"]]) * 255).astype(np.uint8),
                    caption=f"Plot {i+1}",
                    use_container_width=True,
                )
                indices, _ = compute_spectral_indices(crop_i)
                st.metric("NDVI", f"{indices['NDVI']:.2f}")
                st.metric("NDWI", f"{indices['NDWI']:.2f}")

        st.markdown("---")
        st.subheader("📊 Spectral Heatmaps")
        for i, (c, _) in enumerate(plots):
            _, maps = compute_spectral_indices(c)
            for idx_name, arr in maps.items():
                fig = px.imshow(arr, color_continuous_scale="RdYlGn",
                                title=f"Plot {i+1} {idx_name}", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Comparison Table")
        comp_rows = []
        for i, (c, s) in enumerate(plots):
            indices, _ = compute_spectral_indices(c)
            comp_rows.append({
                "Plot": f"Plot {i+1}",
                "NDVI": round(indices["NDVI"], 3),
                "NDWI": round(indices["NDWI"], 3),
                "SAVI": round(indices["SAVI"], 3),
                "GCI": round(indices["GCI"], 3),
                "Avg Moisture": round(float(s["SoilMoisture(%)"].mean()), 1),
                "Avg pH": round(float(s["pH"].mean()), 2),
            })
        st.dataframe(pd.DataFrame(comp_rows))

    # -------------------------
    # Tab 2: Device Data
    # -------------------------
    with tabs[1]:
        st.markdown("### 🛰 Linked Devices")

        if "linked_devices" in st.session_state and st.session_state["linked_devices"]:
            st.dataframe(pd.DataFrame(st.session_state["linked_devices"]))
        else:
            st.info("No devices linked yet.")

    # -------------------------
    # Tab 3: Advisory
    # -------------------------
    with tabs[2]:
        st.markdown("### 📝 Send Advisory to Farmers")

        with st.form("advisory_form"):
            msg = st.text_area("Advisory Message", placeholder="Type your recommendation...")
            file = st.file_uploader("Attach file (optional)", type=["pdf", "png", "jpg", "csv"])
            if st.form_submit_button("Send Advisory"):
                utils.add_chat_entry("Agronomist", message=msg,
                                     file_name=(file.name if file else None),
                                     file_bytes=(file.getvalue() if file else None))
                st.success("✅ Advisory sent successfully!")

    # -------------------------
    # Tab 4: Chat + File + Video
    # -------------------------
    with tabs[3]:
        st.markdown("### 💬 Chat with Farmer")

        reply = st.text_input("Enter reply message")
        if st.button("Send Reply"):
            utils.add_chat_entry("Agronomist", message=reply)
            st.success("✅ Message sent!")

        # Show chat history
        for entry in st.session_state.chat_history[::-1]:
            st.markdown(f"**{entry['sender']}** — {entry['time']}")
            st.write(entry['message'])
            if entry.get("file_name"):
                b64 = base64.b64encode(entry["file_bytes"]).decode()
                st.markdown(f"<a href='data:application/octet-stream;base64,{b64}' "
                            f"download='{entry['file_name']}'>📥 Download {entry['file_name']}</a>",
                            unsafe_allow_html=True)
            st.markdown("---")

        # File Sharing
        st.subheader("📂 Share File with Farmer")
        uploaded_file = st.file_uploader("Upload file", type=None)
        if uploaded_file:
            utils.add_chat_entry("Agronomist",
                                 message=f"📂 Shared a file: {uploaded_file.name}",
                                 file_name=uploaded_file.name,
                                 file_bytes=uploaded_file.read())
            st.success(f"✅ File shared: {uploaded_file.name}")

        # Video Call
        st.subheader("📹 Video Call")
        st.info("Start a live video call with Farmer.")
        webrtc_streamer(
            key="farmer-call",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
