# pages/researcher.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from services import data_generators, model_utils


def app():
    st.header(f"🧑‍🔬 Researcher Dashboard — Welcome, {st.session_state['username']}")
    st.caption("Experiment with ML/DL models on synthetic agriculture datasets 📊")

    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    st.sidebar.subheader("⚙️ Training Settings")
    model_type = st.sidebar.selectbox("Model Type", ["RandomForest", "CNN (image)", "LSTM (time-series)"])
    seed = st.sidebar.number_input("Training Seed", min_value=1, max_value=9999, value=777)
    test_size = st.sidebar.slider("Validation Split", 0.05, 0.5, 0.2)

    if model_type == "RandomForest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 500, 80)
        max_depth = st.sidebar.slider("max_depth (0=None)", 0, 50, 12)
        use_cv = st.sidebar.checkbox("5-fold Cross-validation", value=True)

    epochs = st.sidebar.slider("Epochs (CNN/LSTM)", 1, 50, 5) if model_type != "RandomForest" else None
    batch_size = st.sidebar.slider("Batch Size (CNN/LSTM)", 8, 128, 32) if model_type != "RandomForest" else None

    # -------------------------------
    # Preview synthetic dataset
    # -------------------------------
    st.markdown("### 📊 Dataset Preview")
    demo = data_generators.generate_crop_image(seed=seed)

    rgb = np.dstack([demo["RED"], demo["GREEN"], demo["BLUE"]])
    col1, col2 = st.columns(2)
    with col1:
        st.image((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                 caption="Synthetic RGB Sample",
                 use_container_width=True)
    with col2:
        fig = px.imshow(demo["NDVI"], color_continuous_scale="RdYlGn", title="Synthetic NDVI")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # Training
    # -------------------------------
    if st.button("🚀 Train & Evaluate Model", use_container_width=True):
        st.info("⏳ Preparing dataset and training model...")

        demo = data_generators.generate_crop_image(seed=seed)
        model_obj, acc_final = None, None

        if model_type == "RandomForest":
            X, y = demo["X"], demo["y"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(seed),
                stratify=y if len(np.unique(y)) > 1 else None
            )
            clf = model_utils.train_crop_model(X_train, y_train,
                                               seed=seed,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.metric("Validation Accuracy", f"{acc:.3f}")
            st.subheader("📑 Classification Report")
            st.text(classification_report(y_test, y_pred, digits=3))

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Healthy", "Stressed"])
            ax_cm.set_yticklabels(["Healthy", "Stressed"])
            for (i, j), val in np.ndenumerate(cm):
                ax_cm.text(j, i, int(val), ha="center", va="center", color="black")
            st.pyplot(fig_cm)

            st.subheader("🌟 Feature Importances")
            fi_df = pd.DataFrame({
                "Feature": ["RED", "GREEN", "BLUE", "NIR", "NDVI"],
                "Importance": clf.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.bar_chart(fi_df.set_index("Feature"))

            if use_cv:
                with st.spinner("Running 5-fold Cross-validation..."):
                    scores = cross_val_score(clf, X, y, cv=5)
                st.success(f"✅ CV mean accuracy: {scores.mean():.3f}")

            model_obj, acc_final = clf, float(acc)

        elif model_type == "CNN (image)":
            X = np.dstack([demo["RED"], demo["GREEN"], demo["BLUE"], demo["NIR"]])
            y = demo["labels"]
            X = np.expand_dims(X, axis=0)
            y = np.expand_dims(y, axis=0)

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=X.shape[1:]),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax")
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            acc_final = history.history["accuracy"][-1]
            st.metric("Training Accuracy (demo)", f"{acc_final:.3f}")
            model_obj = model

        elif model_type == "LSTM (time-series)":
            soil = data_generators.generate_soil_data(days=50, seed=seed)
            features = soil[["SoilMoisture(%)", "SoilTemp(C)", "pH"]].values
            X, y = [], []
            window = 5
            for i in range(len(features) - window):
                X.append(features[i:i + window])
                y.append(0 if soil["SoilMoisture(%)"].iloc[i + window] > 40 else 1)
            X, y = np.array(X), np.array(y)

            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, input_shape=(X.shape[1], X.shape[2])),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax")
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            acc_final = history.history["accuracy"][-1]
            st.metric("Training Accuracy (demo)", f"{acc_final:.3f}")
            model_obj = model

        # -------------------------------
        # Save trained model
        # -------------------------------
        buf = io.BytesIO()
        pickle.dump(model_obj, buf)
        buf.seek(0)
        st.download_button(
            "📥 Download Trained Model",
            data=buf.getvalue(),
            file_name=f"agroniq_{model_type.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream",
        )

        st.session_state["researcher_last_model"] = {
            "model": model_obj,
            "type": model_type,
            "seed": int(seed),
            "test_size": float(test_size),
            "accuracy": float(acc_final),
        }
        st.success("✅ Training complete & snapshot saved!")

    # -------------------------------
    # Snapshot of last model
    # -------------------------------
    if st.session_state.get("researcher_last_model"):
        st.markdown("---")
        st.subheader("📌 Last Trained Model (Snapshot)")
        info = st.session_state["researcher_last_model"]
        st.write(f"Type: **{info['type']}**, Seed: **{info['seed']}**, Test split: **{info['test_size']}**")
        st.metric("Last Training Accuracy", f"{info['accuracy']:.3f}")

        try:
            buf = io.BytesIO()
            pickle.dump(info["model"], buf)
            buf.seek(0)
            st.download_button(
                "📥 Download Last Model",
                data=buf.getvalue(),
                file_name="agroniq_last_model.pkl",
                mime="application/octet-stream",
            )
        except Exception:
            st.info("ℹ️ Model artifact unavailable for download.")
