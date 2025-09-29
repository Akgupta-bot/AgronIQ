# pages/farmer.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from PIL import Image
from services import data_generators, model_utils, report_generator, utils
from streamlit_webrtc import webrtc_streamer, WebRtcMode


def app():
    st.header(f"🌾 Farmer Dashboard — Welcome, {st.session_state['username']}")

    st.markdown(
        "Easily upload crop images, soil & weather data, detect anomalies, "
        "get AI-powered recommendations, generate reports, and connect with Agronomists."
    )

    # --------------------------
    # Session state init
    # --------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # --------------------------
    # Tabs
    # --------------------------
    tabs = st.tabs(
        [
            "📸 Crop Analysis",
            "🌱 Soil & Weather",
            "⚠️ Anomalies",
            "📊 Recommendations",
            "📄 Report",
            "💬 Chat",
        ]
    )

    # --------------------------------
    # Tab 1: Crop Analysis
    # --------------------------------
    with tabs[0]:
        st.subheader("📸 Crop Image Analysis")

        uploaded_image = st.file_uploader(
            "Upload Crop Image (RGB or multispectral)", type=["jpg", "jpeg", "png"]
        )

        seed_input = st.number_input(
            "Seed (demo data)", min_value=1, max_value=9999, value=42
        )

        model_choice = st.radio("Select Model", ["RandomForest", "CNN"], horizontal=True)

        clf, crop_sample = None, None
        try:
            clf, crop_sample = model_utils.train_model(seed=seed_input)
        except Exception:
            st.warning("Could not initialize baseline model.")

        crop_image_bytes = None
        final_recommendations = []

        if st.button("▶️ Run Crop Analysis"):
            if uploaded_image:
                try:
                    img = Image.open(uploaded_image).convert("RGB")
                    img_resized = img.resize((128, 128))
                    crop_image_bytes = uploaded_image.getvalue()

                    arr = np.array(img_resized) / 255.0
                    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

                    # Simulated extra bands
                    nir = red * 0.6 + green * 0.2
                    red_edge = (green * 0.3 + nir * 0.7)
                    swir = (red * 0.2 + nir * 0.8)

                    # Spectral indices
                    ndvi = (nir - red) / (nir + red + 1e-6)
                    ndwi = (green - nir) / (green + nir + 1e-6)
                    savi = (1.5 * (nir - red)) / (nir + red + 0.5 + 1e-6)
                    gci = (nir / (green + 1e-6)) - 1

                    multispectral = {
    "red": red, "green": green, "blue": blue,
    "nir": nir, "red_edge": red_edge, "swir": swir,
    "ndvi": ndvi, "ndwi": ndwi, "savi": savi, "gci": gci,
}


                    # Predictions
                    if model_choice == "RandomForest" and clf:
                        preds = model_utils.predict_stress_map(clf, multispectral)
                    elif model_choice == "CNN":
                        if model_utils.TF_AVAILABLE:
                            model_utils.train_cnn(multispectral, epochs=2)
                            preds = np.random.randint(0, 2, size=red.shape)  # placeholder
                        else:
                            st.error("TensorFlow not installed, CNN unavailable.")
                            preds = np.zeros_like(red)
                    else:
                        preds = np.zeros_like(red)

                    utils.plot_crop_analysis(arr, ndvi, preds)

                    health_pct = 100.0 * (1.0 - preds.mean())
                    stress_pct = 100.0 * preds.mean()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("🌱 Estimated Healthy Area", f"{health_pct:.1f}%")
                    with c2:
                        st.metric("⚠️ Estimated Stressed Area", f"{stress_pct:.1f}%")

                    if stress_pct > 25:
                        final_recommendations.append(
                            "🚨 Significant stressed area detected — schedule in-field scouting and consider treatment."
                        )
                    else:
                        final_recommendations.append(
                            "✅ Crop largely healthy — continue routine monitoring."
                        )

                    st.session_state.analysis_results = {
                        "crop_image_bytes": crop_image_bytes,
                        "soil_df": None,
                        "weather_df": None,
                        "pest_df": None,
                        "recommendations": final_recommendations,
                    }

                except Exception as e:
                    st.error(f"Image processing failed: {e}")
            else:
                st.info("No image uploaded — showing demo data.")

                try:
                    # ✅ FIX: Generate demo crop data properly
                    demo_crop = data_generators.generate_crop_image(seed=seed_input)

                    rgb = np.dstack([demo_crop["red"], demo_crop["green"], demo_crop["blue"]])
                    ndvi = demo_crop["ndvi"]

                    if clf:
                        preds = clf.predict(demo_crop["X"]).reshape(ndvi.shape)
                    else:
                        preds = np.zeros_like(ndvi)

                    utils.plot_crop_analysis(rgb, ndvi, preds)

                except Exception as e:
                    st.error(f"Demo data generation failed: {e}")

    # --------------------------------
    # Tab 2: Soil & Weather
    # --------------------------------
    with tabs[1]:
        st.subheader("🌱 Soil & Weather Data")

        uploaded_soil = st.file_uploader("Upload Soil CSV", type=["csv"])
        default_soil = data_generators.generate_soil_data(seed=seed_input)
        default_weather = data_generators.generate_weather_data(seed=seed_input)

        if uploaded_soil:
            try:
                raw_df = pd.read_csv(uploaded_soil)
                soil_df = utils.standardize_soil_df(raw_df)
            except Exception as e:
                st.error(f"Failed to read soil CSV: {e}")
                soil_df = utils.standardize_soil_df(default_soil)
        else:
            soil_df = utils.standardize_soil_df(default_soil)

        st.dataframe(soil_df.head(), use_container_width=True)
        st.plotly_chart(utils.plot_soil_trends(soil_df), use_container_width=True)
        st.plotly_chart(utils.plot_soil_nutrients(soil_df), use_container_width=True)

        weather = default_weather
        st.dataframe(weather.head(), use_container_width=True)

        pest_scores, reasons = [], []
        for i in range(len(weather)):
            sm = soil_df.iloc[i % len(soil_df)]["SoilMoisture(%)"]
            score, reason = utils.pest_risk_score(weather.iloc[i], sm)
            pest_scores.append(score)
            reasons.append("; ".join(reason) if reason else "None")

        weather["PestScore"] = pest_scores
        weather["Reasons"] = reasons
        weather["RiskLevel"] = [utils.pest_risk_level(s)[0] for s in pest_scores]
        weather["Symbol"] = [utils.pest_risk_level(s)[1] for s in pest_scores]

        st.dataframe(weather.head(), use_container_width=True)
        latest = weather.iloc[-1]
        st.info(f"📊 Latest Pest Risk: **{latest['RiskLevel']}** ({latest['Reasons']})")

        if st.session_state.analysis_results:
            st.session_state.analysis_results.update(
                {"soil_df": soil_df, "weather_df": weather, "pest_df": weather}
            )

    # --------------------------------
    # Tab 3: Anomaly Detection
    # --------------------------------
    with tabs[2]:
        st.subheader("⚠️ Anomaly Detection in Soil & Weather")

        if st.session_state.analysis_results and st.session_state.analysis_results["soil_df"] is not None:
            soil_df = st.session_state.analysis_results["soil_df"]
            weather = st.session_state.analysis_results["weather_df"]

            st.markdown("### 🌱 Soil anomalies")
            for col in ["SoilMoisture(%)", "SoilTemp(C)", "pH"]:
                if col in soil_df.columns:
                    zscores = (soil_df[col] - soil_df[col].mean()) / (soil_df[col].std() + 1e-6)
                    anomalies = soil_df[abs(zscores) > 2]
                    if not anomalies.empty:
                        st.warning(f"⚠️ {col} anomalies detected")
                        st.dataframe(anomalies)

            st.markdown("### 🌦️ Weather anomalies")
            for col in ["Rainfall(mm)", "Temp(C)", "Humidity(%)"]:
                if col in weather.columns:
                    zscores = (weather[col] - weather[col].mean()) / (weather[col].std() + 1e-6)
                    w_anomalies = weather[abs(zscores) > 2]
                    if not w_anomalies.empty:
                        st.warning(f"⚠️ {col} anomalies detected")
                        st.dataframe(w_anomalies)
        else:
            st.info("📥 Upload data and run analysis first.")

    # --------------------------------
    # Tab 4: Recommendations
    # --------------------------------
    with tabs[3]:
        st.subheader("🤖 AI Recommendations")
        if st.session_state.analysis_results:
            recs = st.session_state.analysis_results["recommendations"]
            for r in recs:
                st.success(r)
        else:
            st.info("Run an analysis first to see recommendations.")

    # --------------------------------
    # Tab 5: Report
    # --------------------------------
    with tabs[4]:
        st.subheader("📄 Generate PDF Report")
        if st.session_state.analysis_results:
            if st.button("Generate Report"):
                res = st.session_state.analysis_results
                buf = report_generator.create_pdf_report(
                    res["crop_image_bytes"],
                    res["soil_df"],
                    res["weather_df"],
                    res["pest_df"],
                    res["recommendations"],
                )
                st.download_button(
                    "📥 Download Report",
                    data=buf.getvalue(),
                    file_name="AgronIQ_Report.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("Run an analysis first.")

    # --------------------------------
    # Tab 6: Chat + File Sharing + Video
    # --------------------------------
    with tabs[5]:
        st.subheader("💬 Chat with Agronomist")

        chat_input = st.text_input("✍️ Type your message")
        if st.button("Send"):
            if chat_input.strip():
                utils.add_chat_entry("Farmer", message=chat_input.strip())
                st.success("Message sent")

        st.markdown("### Conversation History")
        for entry in st.session_state.chat_history[::-1]:
            sender, msg, time = entry["sender"], entry["message"], entry["time"]
            st.markdown(f"**{sender}** — {time}")
            st.write(msg)
            if entry.get("file_name"):
                b64 = base64.b64encode(entry["file_bytes"]).decode()
                href = f"<a href='data:application/octet-stream;base64,{b64}' download='{entry['file_name']}'>📥 Download {entry['file_name']}</a>"
                st.markdown(href, unsafe_allow_html=True)
            st.markdown("---")

        st.subheader("📂 File Sharing")
        uploaded_file = st.file_uploader("Send a file to Agronomist", type=None)
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            shared_dir = "shared_files"
            os.makedirs(shared_dir, exist_ok=True)
            with open(os.path.join(shared_dir, file_name), "wb") as f:
                f.write(file_bytes)

            utils.add_chat_entry(
                "Farmer",
                message=f"📂 Shared a file: {file_name}",
                file_name=file_name,
                file_bytes=file_bytes,
            )
            st.success(f"✅ File shared: {file_name}")

        st.subheader("📹 Video Call with Agronomist")
        webrtc_streamer(
            key="farmer-call",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
